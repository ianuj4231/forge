from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


Intent = Literal[
    "refill",
    "reschedule",
    "clinical",
    "new_patient",
    "billing",
    "schedule",
    "records",
    "cancel",
    "admin",
    "third_party",
    "other",
    "spam",
]

Urgency = Literal["emergency", "urgent", "routine"]


class MessageInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    received_at: str | None = None
    channel: str | None = None
    from_name: str = ""
    from_contact: str = ""
    body: str


class MessageClassification(BaseModel):
    intents: list[Intent] = Field(
        default_factory=list,
        description="One or more intents expressed in the message.",
    )
    urgency: Urgency = "routine"


Classification = MessageClassification


class MedicationExtraction(BaseModel):
    requested_medication: str | None = Field(
        default=None,
        description="Medication explicitly requested for refill, or null if none is named.",
    )


class AgentState(TypedDict, total=False):
    execution_id: str
    id: str
    received_at: str | None
    channel: str | None
    from_name: str
    from_contact: str
    message: str
    patient_id: str | None
    verified: bool
    verification_reason: str
    verification_branch: str
    non_emergency_branch: str
    requested_medication: str | None
    pinpointed_item: dict[str, Any] | None
    match_status: str
    intents: list[Intent]
    urgency: Urgency
    patient_record: dict[str, Any] | None
    actions: list[str]
    routed_to: list[str]
    tool_calls: list[str]
    reply_draft: str
    workflow_results: list[dict[str, Any]]
    appointment_result: dict[str, Any] | None
    result: dict[str, Any]
    error: str | None


class ProcessResponse(BaseModel):
    execution_id: str
    id: str
    patient_id: str | None
    intents: list[str]
    urgency: Urgency
    actions: list[str]
    routed_to: list[str]
    tool_calls: list[str]
    reply_draft: str
