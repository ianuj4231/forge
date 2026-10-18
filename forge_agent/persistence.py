from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    state_json TEXT,
                    error TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def start(self, execution_id: str, message_id: str, payload: dict[str, Any]) -> None:
        logger.info(
            "execution persistence started execution_id=%s message_id=%s",
            execution_id,
            message_id,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO executions (execution_id, message_id, status, started_at, input_json) VALUES (?, ?, ?, ?, ?)",
                (execution_id, message_id, "running", utc_now(), json.dumps(payload, ensure_ascii=False)),
            )

    def complete(self, execution_id: str, state: dict[str, Any]) -> None:
        result = state.get("result") or public_result(state)
        logger.info("execution persistence completed execution_id=%s", execution_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE executions SET status=?, completed_at=?, result_json=?, state_json=?, error=? WHERE execution_id=?",
                ("completed", utc_now(), json.dumps(result, ensure_ascii=False), json.dumps(state, ensure_ascii=False), state.get("error"), execution_id),
            )

    def fail(self, execution_id: str, error: str, state: dict[str, Any] | None = None) -> None:
        logger.error("execution persistence failed execution_id=%s", execution_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE executions SET status=?, completed_at=?, state_json=?, error=? WHERE execution_id=?",
                ("failed", utc_now(), json.dumps(state or {}, ensure_ascii=False), error, execution_id),
            )


def public_result(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": state.get("id"),
        "patient_id": state.get("patient_id"),
        "intents": state.get("intents", []),
        "urgency": state.get("urgency", "routine"),
        "actions": state.get("actions", []),
        "routed_to": state.get("routed_to", []),
        "tool_calls": state.get("tool_calls", []),
        "reply_draft": state.get("reply_draft", ""),
    }
