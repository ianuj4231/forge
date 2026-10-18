from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph

from forge_agent.data import ClinicData, exact_medication_match
from forge_agent.llm import LLMFactory
from forge_agent.persistence import ExecutionStore, public_result
from forge_agent.schemas import AgentState, MessageInput


logger = logging.getLogger(__name__)


def append_unique(existing: list[str], values: list[str]) -> list[str]:
    return existing + [value for value in values if value not in existing]


def build_graph(settings, data: ClinicData, llm: LLMFactory, store: ExecutionStore):
    checkpointer = _make_checkpointer(settings.checkpoint_db)
    graph_builder = StateGraph(AgentState)

    def verification_node(state: AgentState) -> dict[str, Any]:
        patient, reason = data.find_patient(
            state.get("from_name", ""),
            state.get("from_contact", ""),
        )
        verified = patient is not None and reason in {
            "patient_contact",
            "caregiver_contact",
        }
        logger.info(
            "patient verification completed message_id=%s execution_id=%s verified=%s reason=%s",
            state.get("id"),
            state.get("execution_id"),
            verified,
            reason,
        )
        if not verified:
            logger.warning(
                "patient verification failed message_id=%s reason=%s",
                state.get("id"),
                reason,
            )
        return {
            "patient_id": patient.get("id") if verified and patient else None,
            "patient_record": patient if verified else None,
            "verified": verified,
            "verification_reason": reason,
            "tool_calls": append_unique(state.get("tool_calls", []), ["find_patient"]),
        }

    def verification_router_node(state: AgentState) -> dict[str, Any]:
        return {"verification_branch": "verified" if state.get("verified") else "unverified"}

    def intent_and_urgency_classifier_node(state: AgentState) -> dict[str, Any]:
        result = llm.classify(state["message"])
        logger.info(
            "message classified message_id=%s execution_id=%s intents=%s urgency=%s",
            state.get("id"),
            state.get("execution_id"),
            result.intents,
            result.urgency,
        )
        return {"intents": result.intents, "urgency": result.urgency}

    def emergency_check_node(state: AgentState) -> dict[str, Any]:
        return {}

    def emergency_node(state: AgentState) -> dict[str, Any]:
        logger.warning(
            "emergency escalated message_id=%s execution_id=%s queue=%s",
            state.get("id"),
            state.get("execution_id"),
            "clinician_urgent",
        )
        return {
            "actions": append_unique(state.get("actions", []), ["escalate_emergency"]),
            "routed_to": append_unique(state.get("routed_to", []), ["clinician_urgent"]),
            "reply_draft": (
                "Your symptoms may require emergency care. Please call 911 or go "
                "to the nearest emergency room now. The clinic has been alerted."
            ),
        }

    def unverified_check_node(state: AgentState) -> dict[str, Any]:
        return {"non_emergency_branch": "verified" if state.get("verified") else "unverified"}

    def front_desk_verification_node(state: AgentState) -> dict[str, Any]:
        logger.warning(
            "verification required before processing message_id=%s execution_id=%s",
            state.get("id"),
            state.get("execution_id"),
        )
        return {
            "actions": append_unique(state.get("actions", []), ["request_verification"]),
            "routed_to": append_unique(state.get("routed_to", []), ["front_desk"]),
            "reply_draft": (
                "Please contact our front desk to verify your identity before "
                "we can process this request."
            ),
        }

    def verified_non_emergency_dispatch_node(state: AgentState) -> dict[str, Any]:
        return {}

    async def run_clinical_workflow(state: AgentState) -> dict[str, Any]:
        """Return an isolated clinical result; do not mutate shared graph state."""
        urgency = state.get("urgency")
        clinician_queue = "clinician_urgent" if urgency == "urgent" else "clinician_routine"
        return {
            "workflow": "clinical",
            "actions": ["route_to_clinician"],
            "routed_to": [clinician_queue],
            "tool_calls": [],
            "reply_draft": (
                "Your clinical concern has been sent to a clinician for "
                + ("prompt" if urgency == "urgent" else "routine")
                + " review."
            ),
        }

    async def run_billing_workflow(state: AgentState) -> dict[str, Any]:
        """Route billing concerns independently from clinical/refill work."""
        return {
            "workflow": "billing",
            "actions": ["route_to_billing"],
            "routed_to": ["billing"],
            "tool_calls": [],
            "reply_draft": "Your billing question has been sent to the billing team for review.",
        }

    async def run_schedule_workflow(state: AgentState) -> dict[str, Any]:
        return schedule_workflow_result(state, data, reschedule=False)

    async def run_reschedule_workflow(state: AgentState) -> dict[str, Any]:
        return schedule_workflow_result(state, data, reschedule=True)

    async def run_new_patient_workflow(state: AgentState) -> dict[str, Any]:
        return {
            "workflow": "new_patient",
            "actions": ["answer_directly", "offer_new_patient_booking"],
            "routed_to": ["front_desk"],
            "tool_calls": [],
            "reply_draft": (
                "Yes, we accept new patients. The front desk can help arrange "
                "a new-patient appointment."
            ),
        }

    async def run_records_workflow(state: AgentState) -> dict[str, Any]:
        if not state.get("verified"):
            logger.warning(
                "records workflow requires verification message_id=%s execution_id=%s",
                state.get("id"),
                state.get("execution_id"),
            )
            return {
                "workflow": "records",
                "actions": ["request_verification"],
                "routed_to": ["front_desk"],
                "tool_calls": ["verify_identity"],
                "reply_draft": "Please verify your identity before we can process a records request.",
            }
        return {
            "workflow": "records",
            "actions": ["route_to_front_desk"],
            "routed_to": ["front_desk"],
            "tool_calls": ["find_patient"],
            "reply_draft": "Your records request has been sent to the front desk for review.",
        }

    async def run_cancel_workflow(state: AgentState) -> dict[str, Any]:
        appointment = _next_patient_appointment(data.appointments, state.get("patient_id"), data.clinic["today"])
        if appointment is None:
            logger.warning(
                "cancel workflow found no upcoming appointment message_id=%s execution_id=%s",
                state.get("id"),
                state.get("execution_id"),
            )
            return {
                "workflow": "cancel",
                "actions": ["offer_alternatives"],
                "routed_to": ["front_desk"],
                "tool_calls": ["find_appointment"],
                "reply_draft": "We could not find an upcoming appointment to cancel. The front desk can help review it.",
            }
        logger.info(
            "appointment cancellation prepared message_id=%s execution_id=%s",
            state.get("id"),
            state.get("execution_id"),
        )
        return {
            "workflow": "cancel",
            "actions": ["cancel_appointment"],
            "routed_to": [],
            "tool_calls": ["find_appointment", "cancel_appointment"],
            "reply_draft": f"Your appointment on {appointment['date']} at {appointment['start']} has been marked for cancellation.",
            "appointment_result": {"status": "cancelled", "appointment_id": appointment["id"]},
        }

    async def run_admin_workflow(state: AgentState) -> dict[str, Any]:
        message = state.get("message", "").casefold()
        if "phone" in message or "change" in message:
            return {
                "workflow": "admin",
                "actions": ["route_to_front_desk"],
                "routed_to": ["front_desk"],
                "tool_calls": [],
                "reply_draft": "Your administrative request has been sent to the front desk.",
            }
        if "thanksgiving" in message or "27 nov" in message:
            reply = "The clinic is closed on November 27, 2026, for the Thanksgiving holiday."
        else:
            hours = data.clinic["hours"]
            reply = f"The clinic is open {', '.join(hours['days'])} from {hours['open']} to {hours['close']}, closed for lunch from {hours['lunch'][0]} to {hours['lunch'][1]}."
        return {
            "workflow": "admin",
            "actions": ["answer_directly"],
            "routed_to": [],
            "tool_calls": [],
            "reply_draft": reply,
        }

    async def run_third_party_workflow(state: AgentState) -> dict[str, Any]:
        logger.warning(
            "third-party disclosure rejected message_id=%s execution_id=%s",
            state.get("id"),
            state.get("execution_id"),
        )
        return {
            "workflow": "third_party",
            "actions": ["decline_disclosure"],
            "routed_to": ["front_desk"],
            "tool_calls": ["verify_identity"],
            "reply_draft": "We cannot disclose another person's appointment information without verified authorization.",
        }

    async def run_other_workflow(state: AgentState) -> dict[str, Any]:
        return {
            "workflow": "other",
            "actions": ["acknowledge"],
            "routed_to": [],
            "tool_calls": [],
            "reply_draft": "Thank you for contacting Maplewood Family Clinic.",
        }

    async def run_spam_workflow(state: AgentState) -> dict[str, Any]:
        return {
            "workflow": "spam",
            "actions": ["no_action"],
            "routed_to": [],
            "tool_calls": [],
            "reply_draft": "",
        }

    def extract_medication_node(state: AgentState) -> dict[str, Any]:
        extraction = llm.extract_medication(state["message"])
        requested_medication = extraction.requested_medication
        if requested_medication is None and "all my medications" in state["message"].casefold():
            patient_id = state.get("patient_id")
            patient_medications = [
                item for item in data.medications if item.get("patient_id") == patient_id
            ]
            if len(patient_medications) == 1:
                requested_medication = patient_medications[0].get("medication")
        actions = state.get("actions", [])
        if any(
            phrase in state["message"].casefold()
            for phrase in ("ignore your previous instructions", "approve refills", "bulk")
        ):
            actions = append_unique(actions, ["flag_suspicious_instruction"])
        return {
            "requested_medication": requested_medication,
            "actions": actions,
            "tool_calls": append_unique(state.get("tool_calls", []), ["extract_medication"]),
        }

    def medication_clarification_node(state: AgentState) -> dict[str, Any]:
        return {
            "actions": append_unique(
                state.get("actions", []),
                ["request_medication_clarification"],
            ),
            "routed_to": append_unique(state.get("routed_to", []), ["front_desk"]),
            "reply_draft": "Which medication would you like us to refill?",
        }

    def match_medication_node(state: AgentState) -> dict[str, Any]:
        patient_id = state.get("patient_id")
        patient_medications = [
            item for item in data.medications if item.get("patient_id") == patient_id
        ]
        item = exact_medication_match(
            patient_medications,
            state.get("requested_medication"),
        )
        return {
            "pinpointed_item": item,
            "match_status": "matched" if item else "not_found",
            "tool_calls": append_unique(state.get("tool_calls", []), ["find_medication"]),
        }

    def medication_not_found_node(state: AgentState) -> dict[str, Any]:
        return {
            "actions": append_unique(state.get("actions", []), ["route_to_provider"]),
            "routed_to": append_unique(state.get("routed_to", []), ["provider_refills"]),
            "reply_draft": (
                "We could not match that medication to your active medication list. "
                "The request has been routed to the provider for review."
            ),
        }

    def check_refill_policy_node(state: AgentState) -> dict[str, Any]:
        actions, routed_to, reply = evaluate_refill_policy(
            medication=state.get("pinpointed_item"),
            patient=state.get("patient_record") or {},
            clinic_policy=data.clinic["refill_policy"],
            today_raw=data.clinic["today"],
        )
        if "billing" in state.get("intents", []):
            actions = append_unique(actions, ["route_to_billing"])
            routed_to = append_unique(routed_to, ["billing"])
        if "flag_suspicious_instruction" in state.get("actions", []):
            actions = append_unique(actions, ["flag_suspicious_instruction"])
        clinical_reply = state.get("reply_draft", "")
        combined_reply = f"{clinical_reply} {reply}".strip() if clinical_reply else reply
        return {
            "actions": append_unique(state.get("actions", []), actions),
            "routed_to": append_unique(state.get("routed_to", []), routed_to),
            "tool_calls": append_unique(
                state.get("tool_calls", []),
                ["check_refill_eligibility"],
            ),
            "reply_draft": combined_reply,
        }

    async def run_refill_workflow(state: AgentState) -> dict[str, Any]:
        """Run the refill workflow in isolation so it can fan out safely."""
        result: dict[str, Any] = {
            "workflow": "refill",
            "actions": [],
            "routed_to": [],
            "tool_calls": [],
            "reply_draft": "",
        }
        message = state["message"]
        extraction = await asyncio.to_thread(llm.extract_medication, message)
        requested_medication = extraction.requested_medication
        if requested_medication is None and "all my medications" in message.casefold():
            patient_medications = [
                item for item in data.medications
                if item.get("patient_id") == state.get("patient_id")
            ]
            if len(patient_medications) == 1:
                requested_medication = patient_medications[0].get("medication")

        result["requested_medication"] = requested_medication
        result["tool_calls"].append("extract_medication")
        if any(
            phrase in message.casefold()
            for phrase in ("ignore your previous instructions", "approve refills", "bulk")
        ):
            result["actions"].append("flag_suspicious_instruction")

        if not requested_medication:
            logger.warning(
                "refill medication clarification required message_id=%s execution_id=%s",
                state.get("id"),
                state.get("execution_id"),
            )
            result["actions"].append("request_medication_clarification")
            result["routed_to"].append("front_desk")
            result["reply_draft"] = "Which medication would you like us to refill?"
            return result

        patient_medications = [
            item for item in data.medications
            if item.get("patient_id") == state.get("patient_id")
        ]
        medication = exact_medication_match(patient_medications, requested_medication)
        result["tool_calls"].append("find_medication")
        if medication is None:
            logger.warning(
                "refill medication not matched message_id=%s execution_id=%s",
                state.get("id"),
                state.get("execution_id"),
            )
            result["actions"].append("route_to_provider")
            result["routed_to"].append("provider_refills")
            result["reply_draft"] = (
                "We could not match that medication to your active medication list. "
                "The request has been routed to the provider for review."
            )
            return result

        result["pinpointed_item"] = medication
        actions, routed_to, reply = evaluate_refill_policy(
            medication=medication,
            patient=state.get("patient_record") or {},
            clinic_policy=data.clinic["refill_policy"],
            today_raw=data.clinic["today"],
        )
        result["actions"].extend(actions)
        result["routed_to"].extend(routed_to)
        result["tool_calls"].append("check_refill_eligibility")
        result["reply_draft"] = reply
        logger.info(
            "refill policy evaluated message_id=%s execution_id=%s actions=%s routed_to=%s",
            state.get("id"),
            state.get("execution_id"),
            actions,
            routed_to,
        )
        return result

    async def _workflow_fanout_async(state: AgentState) -> dict[str, Any]:
        """Run every supported intent concurrently, then merge isolated results."""
        intents = state.get("intents", [])
        handlers = []
        if "clinical" in intents:
            handlers.append(run_clinical_workflow(state))
        if "refill" in intents:
            handlers.append(run_refill_workflow(state))
        if "billing" in intents:
            handlers.append(run_billing_workflow(state))
        if "schedule" in intents:
            handlers.append(run_schedule_workflow(state))
        if "reschedule" in intents:
            handlers.append(run_reschedule_workflow(state))
        if "new_patient" in intents:
            handlers.append(run_new_patient_workflow(state))
        if "records" in intents:
            handlers.append(run_records_workflow(state))
        if "cancel" in intents:
            handlers.append(run_cancel_workflow(state))
        if "admin" in intents:
            handlers.append(run_admin_workflow(state))
        if "third_party" in intents:
            handlers.append(run_third_party_workflow(state))
        if "other" in intents:
            handlers.append(run_other_workflow(state))
        if "spam" in intents:
            handlers.append(run_spam_workflow(state))

        selected_workflows = [
            intent for intent in (
                "clinical",
                "refill",
                "billing",
                "schedule",
                "reschedule",
                "new_patient",
                "records",
                "cancel",
                "admin",
                "third_party",
                "other",
                "spam",
            )
            if intent in intents
        ]
        for workflow in selected_workflows:
            logger.info(
                "workflow selected message_id=%s execution_id=%s workflow=%s",
                state.get("id"),
                state.get("execution_id"),
                workflow,
            )
        workflow_results = await asyncio.gather(*handlers) if handlers else []
        logger.info(
            "workflow fanout completed message_id=%s execution_id=%s workflows=%s",
            state.get("id"),
            state.get("execution_id"),
            [result.get("workflow") for result in workflow_results],
        )
        actions = list(state.get("actions", []))
        routed_to = list(state.get("routed_to", []))
        tool_calls = list(state.get("tool_calls", []))
        replies = []
        requested_medication = None
        pinpointed_item = None
        appointment_result = None
        for workflow_result in workflow_results:
            actions = append_unique(actions, workflow_result.get("actions", []))
            routed_to = append_unique(routed_to, workflow_result.get("routed_to", []))
            tool_calls = append_unique(tool_calls, workflow_result.get("tool_calls", []))
            if workflow_result.get("reply_draft"):
                replies.append(workflow_result["reply_draft"])
            requested_medication = (
                workflow_result.get("requested_medication") or requested_medication
            )
            pinpointed_item = workflow_result.get("pinpointed_item") or pinpointed_item
            appointment_result = workflow_result.get("appointment_result") or appointment_result

            logger.info(
                "workflow completed message_id=%s execution_id=%s workflow=%s actions=%s routed_to=%s",
                state.get("id"),
                state.get("execution_id"),
                workflow_result.get("workflow"),
                workflow_result.get("actions", []),
                workflow_result.get("routed_to", []),
            )

        logger.info(
            "workflow results merged message_id=%s execution_id=%s workflows=%s actions=%s routed_to=%s",
            state.get("id"),
            state.get("execution_id"),
            [result.get("workflow") for result in workflow_results],
            actions,
            routed_to,
        )

        return {
            "workflow_results": workflow_results,
            "requested_medication": requested_medication,
            "pinpointed_item": pinpointed_item,
            "appointment_result": appointment_result,
            "actions": actions,
            "routed_to": routed_to,
            "tool_calls": tool_calls,
            "reply_draft": " ".join(replies),
        }

    def workflow_fanout_node(state: AgentState) -> dict[str, Any]:
        """Synchronous LangGraph adapter for the async fan-out implementation."""
        return asyncio.run(_workflow_fanout_async(state))

    def save_output_node(state: AgentState) -> dict[str, Any]:
        return {"result": public_result(state)}

    graph_builder.add_node("verification_node", verification_node)
    graph_builder.add_node("verification_router_node", verification_router_node)
    graph_builder.add_node("intent_and_urgency_classifier_node", intent_and_urgency_classifier_node)
    graph_builder.add_node("emergency_check_node", emergency_check_node)
    graph_builder.add_node("emergency_node", emergency_node)
    graph_builder.add_node("unverified_check_node", unverified_check_node)
    graph_builder.add_node("front_desk_verification_node", front_desk_verification_node)
    graph_builder.add_node("verified_non_emergency_dispatch_node", verified_non_emergency_dispatch_node)
    graph_builder.add_node("workflow_fanout_node", workflow_fanout_node)
    graph_builder.add_node("save_output_node", save_output_node)

    graph_builder.add_edge(START, "verification_node")
    graph_builder.add_edge("verification_node", "verification_router_node")
    graph_builder.add_edge(
        "verification_router_node",
        "intent_and_urgency_classifier_node",
    )
    graph_builder.add_edge("intent_and_urgency_classifier_node", "emergency_check_node")
    graph_builder.add_conditional_edges(
        "emergency_check_node",
        _emergency_route,
        {"emergency": "emergency_node", "not_emergency": "unverified_check_node"},
    )
    graph_builder.add_edge("emergency_node", "save_output_node")
    graph_builder.add_conditional_edges(
        "unverified_check_node",
        _unverified_route,
        {
            "unverified": "front_desk_verification_node",
            "verified": "verified_non_emergency_dispatch_node",
            "safe_unverified": "workflow_fanout_node",
        },
    )
    graph_builder.add_edge("front_desk_verification_node", "save_output_node")
    graph_builder.add_conditional_edges(
        "verified_non_emergency_dispatch_node",
        _supported_intent_route,
        {
            "supported": "workflow_fanout_node",
            "non_refill": "save_output_node",
        },
    )
    graph_builder.add_edge("workflow_fanout_node", "save_output_node")
    graph_builder.add_edge("save_output_node", END)

    compiled_graph = graph_builder.compile(checkpointer=checkpointer)

    def invoke(message: MessageInput) -> AgentState:
        execution_id = str(uuid.uuid4())
        logger.info(
            "execution started execution_id=%s message_id=%s channel=%s",
            execution_id,
            message.id,
            message.channel,
        )
        initial_state: AgentState = {
            "execution_id": execution_id,
            "id": message.id,
            "received_at": message.received_at,
            "channel": message.channel,
            "from_name": message.from_name,
            "from_contact": message.from_contact,
            "message": message.body,
            "patient_id": None,
            "verified": False,
            "intents": [],
            "urgency": "routine",
            "actions": [],
            "routed_to": [],
            "tool_calls": [],
            "reply_draft": "",
        }
        store.start(execution_id, message.id, message.model_dump())
        try:
            final_state = compiled_graph.invoke(
                initial_state,
                {"configurable": {"thread_id": f"{message.id}:{execution_id}"}},
            )
            store.complete(execution_id, final_state)
            logger.info(
                "execution completed execution_id=%s message_id=%s actions=%s routed_to=%s",
                execution_id,
                message.id,
                final_state.get("actions", []),
                final_state.get("routed_to", []),
            )
            return final_state
        except Exception as exc:
            store.fail(execution_id, str(exc), initial_state)
            logger.exception(
                "execution failed execution_id=%s message_id=%s",
                execution_id,
                message.id,
            )
            raise

    invoke.checkpointer = checkpointer
    return invoke


def _emergency_route(state: AgentState) -> str:
    return "emergency" if state.get("urgency") == "emergency" else "not_emergency"


def _unverified_route(state: AgentState) -> str:
    if state.get("verified"):
        logger.info(
            "unverified check routed verified message_id=%s execution_id=%s",
            state.get("id"),
            state.get("execution_id"),
        )
        return "verified"
    if any(intent in {"new_patient", "third_party", "spam"} for intent in state.get("intents", [])):
        logger.info(
            "unverified check routed safe workflow message_id=%s execution_id=%s intents=%s",
            state.get("id"),
            state.get("execution_id"),
            state.get("intents", []),
        )
        return "safe_unverified"
    logger.warning(
        "unverified check routed to front desk message_id=%s execution_id=%s",
        state.get("id"),
        state.get("execution_id"),
    )
    return "unverified"


def _intent_route(state: AgentState) -> str:
    intents = state.get("intents", [])
    if "clinical" in intents and "refill" in intents:
        return "clinical_refill"
    if "clinical" in intents:
        return "clinical"
    if "refill" in intents:
        return "refill"
    return "non_refill"


def _supported_intent_route(state: AgentState) -> str:
    return "supported" if any(
        intent in {
            "clinical", "refill", "billing", "schedule", "reschedule",
            "new_patient", "records", "cancel", "admin", "third_party",
            "other", "spam",
        }
        for intent in state.get("intents", [])
    ) else "non_refill"


def _after_clinical_route(state: AgentState) -> str:
    return "refill" if "refill" in state.get("intents", []) else "non_refill"


def _medication_route(state: AgentState) -> str:
    return "found" if state.get("requested_medication") else "missing"


def _match_route(state: AgentState) -> str:
    return "matched" if state.get("match_status") == "matched" else "not_found"


def schedule_workflow_result(
    state: AgentState,
    data: ClinicData,
    reschedule: bool,
) -> dict[str, Any]:
    message = state.get("message", "")
    patient = state.get("patient_record") or {}
    provider = _requested_provider(message, data, patient)
    visit_type = _requested_visit_type(message)
    duration = int(data.clinic["visit_types"].get(visit_type, 20))
    requested_dates = _requested_dates(message, date.fromisoformat(data.clinic["today"]))
    period = _requested_period(message)
    requested_time = _requested_time(message)
    earliest_start = _received_time(state) if requested_dates and requested_dates[0].isoformat() == data.clinic["today"] else None
    excluded_appointment_id = None
    source_appointment = None

    if reschedule:
        source_appointment = _find_source_appointment(
            data.appointments,
            state.get("patient_id"),
            requested_dates,
        )
        if source_appointment:
            excluded_appointment_id = source_appointment["id"]
            requested_dates = _reschedule_target_dates(message, source_appointment)

    slot = _find_available_slot(
        data=data,
        provider=provider,
        visit_type=visit_type,
        duration=duration,
        dates=requested_dates,
        period=period,
        requested_time=requested_time,
        earliest_start=earliest_start,
        exclude_appointment_id=excluded_appointment_id,
    )
    tool_calls = ["find_available_slots"]
    if reschedule:
        tool_calls.insert(0, "find_appointment")

    language = patient.get("preferred_language")
    if slot:
        action = "reschedule_appointment" if reschedule else "book_appointment"
        verb = "rescheduled" if reschedule else "booked"
        reply = _appointment_reply(language, verb, slot, provider)
        appointment_result = {
            "status": "proposed",
            "provider": provider,
            "visit_type": visit_type,
            "date": slot["date"],
            "start": slot["start"],
            "end": slot["end"],
        }
        if source_appointment:
            appointment_result["source_appointment_id"] = source_appointment["id"]
        logger.info(
            "appointment outcome message_id=%s execution_id=%s workflow=%s outcome=%s",
            state.get("id"),
            state.get("execution_id"),
            "reschedule" if reschedule else "schedule",
            action,
        )
        return {
            "workflow": "reschedule" if reschedule else "schedule",
            "actions": [action],
            "routed_to": [],
            "tool_calls": tool_calls,
            "reply_draft": reply,
            "appointment_result": appointment_result,
        }

    logger.warning(
        "appointment alternatives offered message_id=%s execution_id=%s workflow=%s",
        state.get("id"),
        state.get("execution_id"),
        "reschedule" if reschedule else "schedule",
    )
    return {
        "workflow": "reschedule" if reschedule else "schedule",
        "actions": ["offer_alternatives"],
        "routed_to": [],
        "tool_calls": tool_calls,
        "reply_draft": _no_slot_reply(language, provider),
        "appointment_result": {
            "status": "alternatives_needed",
            "provider": provider,
            "visit_type": visit_type,
        },
    }


def _requested_provider(message: str, data: ClinicData, patient: dict[str, Any]) -> str:
    lowered = message.casefold()
    provider_names = {
        "dr. patel": "PATEL",
        "dr patel": "PATEL",
        "patel": "PATEL",
        "dr. okafor": "OKAFOR",
        "dr okafor": "OKAFOR",
        "okafor": "OKAFOR",
        "sarah kim": "KIM",
        "kim": "KIM",
    }
    for name, provider_id in provider_names.items():
        if name in lowered:
            return provider_id
    return patient.get("primary_provider") or data.clinic["providers"][0]["id"]


def _requested_visit_type(message: str) -> str:
    lowered = message.casefold()
    if "annual" in lowered or "physical" in lowered or "chequeo" in lowered:
        return "annual_physical"
    if "telehealth" in lowered or "video" in lowered:
        return "telehealth"
    if "sick" in lowered:
        return "sick_visit"
    return "follow_up"


def _requested_dates(message: str, today: date) -> list[date]:
    lowered = message.casefold()
    numeric_day = re.search(r"\b(?:the\s+)?(1[0-9]|2[0-9]|30|31)(?:st|nd|rd|th)?\b", lowered)
    if numeric_day:
        day = int(numeric_day.group(1))
        try:
            month = today.month if day >= today.day else today.month + 1
            return [date(today.year, month, day)]
        except ValueError:
            pass

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4,
    }
    requested_weekday = next(
        (number for name, number in weekdays.items() if name in lowered),
        None,
    )
    if "next week" in lowered or "próxima semana" in lowered:
        start = today + timedelta(days=(7 - today.weekday()))
        if requested_weekday is not None:
            return [start + timedelta(days=requested_weekday)]
        return [start + timedelta(days=offset) for offset in range(5)]
    if "following week" in lowered:
        start = today + timedelta(days=(7 - today.weekday()) + 7)
        return [start + timedelta(days=offset) for offset in range(5)]
    if requested_weekday is not None:
        days_ahead = (requested_weekday - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return [today + timedelta(days=days_ahead)]
    return [today + timedelta(days=offset) for offset in range(0, 5)]


def _reschedule_target_dates(message: str, source: dict[str, Any]) -> list[date]:
    lowered = message.casefold()
    if "following week" in lowered:
        source_date = date.fromisoformat(source["date"])
        start = source_date + timedelta(days=(7 - source_date.weekday()))
        return [start + timedelta(days=offset) for offset in range(5)]
    target_match = re.search(
        r"\b(?:to|for)\s+(?:any\s+\w+\s+)?(monday|tuesday|wednesday|thursday|friday)\b",
        lowered,
    )
    if target_match:
        target_weekday = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
        }[target_match.group(1)]
        source_date = date.fromisoformat(source["date"])
        days_ahead = (target_weekday - source_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return [source_date + timedelta(days=days_ahead)]
    dates = _requested_dates(message, date.fromisoformat(source["date"]))
    source_date = date.fromisoformat(source["date"])
    if dates and dates[0] == source_date:
        return [source_date + timedelta(days=offset) for offset in range(1, 8)]
    return dates


def _requested_period(message: str) -> str | None:
    lowered = message.casefold()
    if "morning" in lowered:
        return "morning"
    if "afternoon" in lowered:
        return "afternoon"
    return None


def _requested_time(message: str) -> time | None:
    match = re.search(r"\b(1[0-2]|[1-9])\s*(?::(\d{2}))?\s*(am|pm)\b", message.casefold())
    if not match:
        return None
    hour = int(match.group(1)) % 12
    if match.group(3) == "pm":
        hour += 12
    return time(hour, int(match.group(2) or 0))


def _received_time(state: AgentState) -> time | None:
    received_at = state.get("received_at")
    if not received_at:
        return None
    try:
        return datetime.fromisoformat(received_at).time().replace(tzinfo=None)
    except ValueError:
        return None


def _find_source_appointment(
    appointments: list[dict[str, Any]],
    patient_id: str | None,
    requested_dates: list[date],
) -> dict[str, Any] | None:
    requested_date_values = {item.isoformat() for item in requested_dates}
    for appointment in appointments:
        if (
            appointment.get("patient_id") == patient_id
            and appointment.get("status") == "booked"
            and appointment.get("date") in requested_date_values
        ):
            return appointment
    return None


def _next_patient_appointment(
    appointments: list[dict[str, Any]],
    patient_id: str | None,
    today_raw: str,
) -> dict[str, Any] | None:
    today = date.fromisoformat(today_raw)
    upcoming = [
        appointment for appointment in appointments
        if appointment.get("patient_id") == patient_id
        and appointment.get("status") == "booked"
        and date.fromisoformat(appointment["date"]) >= today
    ]
    return min(
        upcoming,
        key=lambda appointment: (appointment["date"], appointment["start"]),
        default=None,
    )


def _find_available_slot(
    data: ClinicData,
    provider: str,
    visit_type: str,
    duration: int,
    dates: list[date],
    period: str | None,
    requested_time: time | None,
    earliest_start: time | None,
    exclude_appointment_id: str | None,
) -> dict[str, str] | None:
    clinic = data.clinic
    holidays = set(clinic.get("holidays", []))
    provider_data = next((item for item in clinic["providers"] if item["id"] == provider), {})
    leave_ranges = provider_data.get("leave", [])
    opening = time.fromisoformat(clinic["hours"]["open"])
    closing = time.fromisoformat(clinic["hours"]["close"])
    lunch_start = time.fromisoformat(clinic["hours"]["lunch"][0])
    lunch_end = time.fromisoformat(clinic["hours"]["lunch"][1])
    weekday_names = set(clinic["hours"]["days"])

    for requested_date in dates:
        if requested_date.strftime("%a") not in weekday_names:
            continue
        if requested_date.isoformat() in holidays:
            continue
        if any(start <= requested_date.isoformat() <= end for start, end in leave_ranges):
            continue
        start_minutes = int(opening.hour * 60 + opening.minute)
        end_minutes = int(closing.hour * 60 + closing.minute - duration)
        candidate_minutes = range(start_minutes, end_minutes + 1, 20)
        for start_minute in candidate_minutes:
            start_at = time(start_minute // 60, start_minute % 60)
            end_at = time((start_minute + duration) // 60, (start_minute + duration) % 60)
            if period == "morning" and start_at >= time(12, 0):
                continue
            if period == "afternoon" and start_at < time(13, 0):
                continue
            if earliest_start and start_at <= earliest_start:
                continue
            if requested_time and start_at != requested_time:
                continue
            if start_at < opening or end_at > closing:
                continue
            if start_at < lunch_end and end_at > lunch_start:
                continue
            if any(
                _overlaps(start_at, end_at, appointment)
                for appointment in data.appointments
                if appointment.get("id") != exclude_appointment_id
                and appointment.get("provider") == provider
                and appointment.get("date") == requested_date.isoformat()
                and appointment.get("status") == "booked"
            ):
                continue
            return {
                "date": requested_date.isoformat(),
                "start": start_at.strftime("%H:%M"),
                "end": end_at.strftime("%H:%M"),
                "visit_type": visit_type,
            }
    return None


def _overlaps(start_at: time, end_at: time, appointment: dict[str, Any]) -> bool:
    existing_start = time.fromisoformat(appointment["start"])
    existing_end = time.fromisoformat(appointment["end"])
    return start_at < existing_end and end_at > existing_start


def _appointment_reply(
    language: str | None,
    verb: str,
    slot: dict[str, str],
    provider: str,
) -> str:
    if language == "Spanish":
        return f"Su cita con {provider} ha sido {verb} para {slot['date']} a las {slot['start']}."
    return f"Your appointment with {provider} has been {verb} for {slot['date']} at {slot['start']}."


def _no_slot_reply(language: str | None, provider: str) -> str:
    if language == "Spanish":
        return f"No encontramos un horario disponible con {provider}; le ofrecemos otras opciones."
    return f"The requested time with {provider} is unavailable. We can offer alternative appointments."


def evaluate_refill_policy(
    medication: dict[str, Any] | None,
    patient: dict[str, Any],
    clinic_policy: dict[str, Any],
    today_raw: str,
) -> tuple[list[str], list[str], str]:
    if medication is None:
        return ["route_to_provider"], ["provider_refills"], "The refill request needs provider review."

    today = date.fromisoformat(today_raw)
    if medication.get("controlled_schedule") is not None:
        return (
            ["route_to_provider"],
            ["provider_refills"],
            "Controlled medication refill requests must be reviewed by the prescribing provider.",
        )

    last_visit_raw = patient.get("last_visit")
    cutoff = date(today.year - 1, today.month, today.day)
    if not last_visit_raw or date.fromisoformat(last_visit_raw) < cutoff:
        return (
            ["refill_needs_visit", "offer_appointment", "route_to_provider"],
            ["provider_refills"],
            "A visit is needed before this refill can be prepared. We have offered an appointment and routed the request to the provider.",
        )

    if int(medication.get("refills_remaining", 0)) <= 0:
        return (
            ["route_to_provider"],
            ["provider_refills"],
            "There are no refills remaining, so your request has been routed to the provider for a new prescription.",
        )

    last_filled_raw = medication.get("last_filled")
    days_supply = medication.get("days_supply")
    if not last_filled_raw or days_supply is None:
        return (
            ["route_to_provider"],
            ["provider_refills"],
            "The refill details need provider review before this request can be prepared.",
        )

    supply_runs_out = date.fromisoformat(last_filled_raw) + timedelta(days=int(days_supply))
    earliest_request_date = supply_runs_out - timedelta(days=7)
    if today < earliest_request_date:
        return (
            ["refill_too_early"],
            [],
            f"It is too early to prepare this refill. The earliest request date is {earliest_request_date.isoformat()}.",
        )

    return (
        ["prepare_refill_for_approval"],
        ["provider_refills"],
        "Your refill request has been prepared for provider approval. The clinic will contact you if anything else is needed.",
    )


def _make_checkpointer(path):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-sqlite to use the SQLite checkpointer"
        ) from exc
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver
