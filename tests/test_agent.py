from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_agent.config import load_settings
from forge_agent.data import ClinicData
from forge_agent.graph import build_graph, evaluate_refill_policy
from forge_agent.llm import LLMFactory, _invoke_with_retry
from forge_agent.persistence import ExecutionStore
from forge_agent.schemas import MessageInput


ROOT = Path(__file__).parents[1]


def make_processor(tmp_path):
    settings = replace(
        load_settings(ROOT / "config.yaml"),
        execution_db=tmp_path / "executions.db",
        checkpoint_db=tmp_path / "checkpoints.db",
        api_key=None,
    )
    return build_graph(
        settings,
        ClinicData(ROOT),
        LLMFactory(settings),
        ExecutionStore(settings.execution_db),
    )


def message(message_id: str) -> MessageInput:
    messages = json.loads(
        (ROOT / "messages_sample.json").read_text(encoding="utf-8-sig")
    )
    return MessageInput.model_validate(
        next(item for item in messages if item["id"] == message_id)
    )


def test_unverified_non_emergency_goes_to_front_desk(tmp_path):
    state = make_processor(tmp_path)(message("M011"))
    assert state["verified"] is False
    assert state["non_emergency_branch"] == "unverified"
    assert state["actions"] == ["request_verification"]
    assert state["routed_to"] == ["front_desk"]


def test_emergency_bypasses_front_desk(tmp_path):
    state = make_processor(tmp_path)(message("M003"))
    assert state["urgency"] == "emergency"
    assert state["actions"] == ["escalate_emergency"]
    assert state["routed_to"] == ["clinician_urgent"]
    assert "911" in state["reply_draft"]


def test_verified_refill_enters_refill_workflow(tmp_path):
    state = make_processor(tmp_path)(message("M001"))
    assert state["verified"] is True
    assert state["non_emergency_branch"] == "verified"
    assert state["actions"] == ["prepare_refill_for_approval"]
    assert state["routed_to"] == ["provider_refills"]
    assert state["requested_medication"] == "lisinopril"
    assert state["pinpointed_item"]["patient_id"] == "P002"


def test_refill_policy_uses_clinic_rules():
    policy = json.loads((ROOT / "clinic.json").read_text(encoding="utf-8"))["refill_policy"]
    medication = {
        "controlled_schedule": "C-IV",
        "refills_remaining": 3,
        "last_filled": "2026-10-01",
        "days_supply": 30,
    }
    actions, routed, _ = evaluate_refill_policy(
        medication,
        {"last_visit": "2026-10-01"},
        policy,
        "2026-11-09",
    )
    assert actions == ["route_to_provider"]
    assert routed == ["provider_refills"]

    old_visit_medication = {**medication, "controlled_schedule": None}
    actions, _, _ = evaluate_refill_policy(
        old_visit_medication,
        {"last_visit": "2025-06-10"},
        policy,
        "2026-11-09",
    )
    assert actions == ["refill_needs_visit", "offer_appointment", "route_to_provider"]

    zero_refills = {**old_visit_medication, "refills_remaining": 0}
    actions, _, _ = evaluate_refill_policy(
        zero_refills,
        {"last_visit": "2026-07-01"},
        policy,
        "2026-11-09",
    )
    assert actions == ["route_to_provider"]


def test_multi_intent_refill_preserves_clinical_and_billing_routes(tmp_path):
    processor = make_processor(tmp_path)
    clinical_state = processor(message("M006"))
    assert "prepare_refill_for_approval" in clinical_state["actions"]
    assert "route_to_clinician" in clinical_state["actions"]
    assert "clinician_urgent" in clinical_state["routed_to"]

    billing_state = processor(message("M025"))
    assert "prepare_refill_for_approval" in billing_state["actions"]
    assert "route_to_billing" in billing_state["actions"]
    assert "billing" in billing_state["routed_to"]


def test_routine_clinical_message_routes_without_automated_advice(tmp_path):
    state = make_processor(tmp_path)(message("M030"))
    assert state["actions"] == ["route_to_clinician"]
    assert state["routed_to"] == ["clinician_routine"]
    assert "automated_clinical_advice" not in state["actions"]
    assert "explain" not in state["reply_draft"].casefold()


def test_prompt_injection_does_not_approve_or_bulk_process(tmp_path):
    state = make_processor(tmp_path)(message("M018"))
    assert "prepare_refill_for_approval" in state["actions"]
    assert "flag_suspicious_instruction" in state["actions"]
    assert "approve_refill" not in state["actions"]
    assert "bulk_approve" not in state["actions"]


def test_parallel_workflow_merge_is_persisted(tmp_path):
    processor = make_processor(tmp_path)
    state = processor(message("M006"))

    assert [item["workflow"] for item in state["workflow_results"]] == [
        "clinical",
        "refill",
    ]
    with ExecutionStore(tmp_path / "executions.db")._connect() as connection:
        row = connection.execute(
            "SELECT status, result_json, state_json FROM executions WHERE execution_id=?",
            (state["execution_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == "completed"
    assert json.loads(row[1])["actions"] == state["actions"]
    assert "workflow_results" in json.loads(row[2])


def test_schedule_and_reschedule_samples(tmp_path):
    processor = make_processor(tmp_path)
    expected_actions = {
        "M002": "reschedule_appointment",
        "M010": "book_appointment",
        "M016": "book_appointment",
        "M024": "book_appointment",
        "M028": "offer_alternatives",
        "M029": "offer_alternatives",
        "M034": "reschedule_appointment",
        "M015": "book_appointment",
    }

    for message_id, expected_action in expected_actions.items():
        state = processor(message(message_id))
        assert expected_action in state["actions"]
        assert "double_book" not in state["actions"]
        assert "book_on_leave" not in state["actions"]

    spanish_state = processor(message("M016"))
    assert "Su cita" in spanish_state["reply_draft"]


def test_remaining_intent_samples(tmp_path):
    processor = make_processor(tmp_path)
    expected_actions = {
        "M007": {"answer_directly", "offer_new_patient_booking"},
        "M011": {"request_verification"},
        "M022": {"cancel_appointment"},
        "M023": {"answer_directly"},
        "M027": {"decline_disclosure"},
        "M031": {"acknowledge"},
        "M033": {"route_to_front_desk"},
        "M037": {"answer_directly"},
        "M038": {"no_action"},
        "M040": {"route_to_front_desk"},
    }

    for message_id, required_actions in expected_actions.items():
        state = processor(message(message_id))
        assert required_actions.issubset(set(state["actions"]))

    new_patient_state = processor(message("M007"))
    assert new_patient_state["patient_id"] is None

    third_party_state = processor(message("M027"))
    assert "disclose_phi" not in third_party_state["actions"]

    spam_state = processor(message("M038"))
    assert spam_state["tool_calls"] == ["find_patient"]
    assert spam_state["patient_id"] is None


def test_llm_retry_uses_six_attempts_and_exponential_backoff(monkeypatch):
    attempts = []
    delays = []

    def flaky_invoke(messages):
        attempts.append(messages)
        if len(attempts) < 6:
            raise RuntimeError("rate limited")
        return "success"

    monkeypatch.setattr("forge_agent.llm.time.sleep", delays.append)
    settings = SimpleNamespace(
        retry_attempts=6,
        retry_initial_backoff_seconds=1,
        retry_max_backoff_seconds=30,
    )

    result = _invoke_with_retry(
        flaky_invoke,
        [("system", "Return the required schema."), ("human", "test")],
        settings,
    )

    assert result == "success"
    assert len(attempts) == 6
    assert delays == [1, 2, 4, 8, 16]
    assert "rate limited" in attempts[1][0][1]


def test_workflow_logs_have_correlation_ids_without_sensitive_message_data(tmp_path, caplog):
    processor = make_processor(tmp_path)
    sample = message("M006")

    with caplog.at_level(logging.INFO):
        state = processor(sample)

    logs = caplog.text
    assert state["execution_id"] in logs
    assert sample.id in logs
    assert sample.body not in logs
    assert sample.from_contact not in logs
    assert "workflow results merged" in logs
    assert "clinical" in logs
    assert "refill" in logs


def test_failed_execution_is_logged_and_persisted(tmp_path, caplog):
    settings = replace(
        load_settings(ROOT / "config.yaml"),
        execution_db=tmp_path / "executions.db",
        checkpoint_db=tmp_path / "checkpoints.db",
        api_key=None,
    )

    class FailingLLM:
        def classify(self, message_text):
            raise RuntimeError("classification unavailable")

    store = ExecutionStore(settings.execution_db)
    processor = build_graph(settings, ClinicData(ROOT), FailingLLM(), store)
    request = MessageInput(
        id="FAIL-001",
        body="routine test message",
        from_name="Test User",
        from_contact="test@example.com",
    )

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        processor(request)

    with store._connect() as connection:
        row = connection.execute(
            "SELECT status, error FROM executions WHERE message_id=?",
            (request.id,),
        ).fetchone()

    assert row[0] == "failed"
    assert "classification unavailable" in row[1]
    assert request.id in caplog.text
    assert "execution failed" in caplog.text
