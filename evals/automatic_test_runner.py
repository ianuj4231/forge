from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge_agent.api import process
from forge_agent.schemas import MessageInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESSAGES = PROJECT_ROOT / "messages_sample.json"
DEFAULT_LABELS = PROJECT_ROOT / "labels_sample.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "post_runs.jsonl"
DEFAULT_DELAY_SECONDS = 5.0


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def join_messages_and_labels(
    messages: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    labels_by_id = {item["id"]: item for item in labels}
    joined = []
    missing_labels = []
    for message in messages:
        label = labels_by_id.get(message.get("id"))
        if label is None:
            missing_labels.append(message.get("id"))
            continue
        joined.append((message, label))
    if missing_labels:
        raise ValueError(f"Missing labels for message ids: {missing_labels}")
    return joined


def evaluate_result(result: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    actual_actions = set(result.get("actions", []))
    expected_actions = set(label.get("expected_actions", []))
    forbidden_actions = set(label.get("must_not", []))
    actual_intents = set(result.get("intents", []))
    expected_intents = set(label.get("intents", []))
    accepted_urgency = set(label.get("accepted_urgency", []))

    missing_actions = sorted(expected_actions - actual_actions)
    forbidden_found = sorted(forbidden_actions & actual_actions)
    missing_intents = sorted(expected_intents - actual_intents)
    urgency_valid = not accepted_urgency or result.get("urgency") in accepted_urgency
    passed = not missing_actions and not forbidden_found and not missing_intents and urgency_valid
    return {
        "passed": passed,
        "missing_actions": missing_actions,
        "forbidden_actions_found": forbidden_found,
        "missing_intents": missing_intents,
        "urgency_valid": urgency_valid,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def write_json_results(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def write_message_result(results_dir: Path, entry_number: int, record: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"message{entry_number}.json"
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def run(
    messages_path: Path = DEFAULT_MESSAGES,
    labels_path: Path = DEFAULT_LABELS,
    output_path: Path = DEFAULT_OUTPUT,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    start_entry: int = 1,
    end_entry: int | None = None,
    results_dir: Path | None = None,
) -> dict[str, int]:
    messages = load_json(messages_path)
    labels = load_json(labels_path)
    joined = join_messages_and_labels(messages, labels)
    if start_entry < 1:
        raise ValueError("start_entry must be at least 1")
    selected = joined[start_entry - 1:end_entry]
    if not selected:
        raise ValueError("The requested entry range is empty")

    summary = {"total": 0, "passed": 0, "failed": 0}
    json_records: list[dict[str, Any]] = []

    for index, (message_payload, label) in enumerate(selected):
        entry_number = start_entry + index
        request = MessageInput.model_validate(message_payload)
        response = process(request)
        result = response.model_dump(mode="json")
        evaluation = evaluate_result(result, label)
        record = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "id": message_payload["id"],
            "request": message_payload,
            "expected": label,
            "actual": result,
            "evaluation": evaluation,
        }
        if output_path.suffix.lower() == ".json":
            json_records.append(record)
            write_json_results(output_path, json_records)
        else:
            append_jsonl(output_path, record)
        if results_dir is not None:
            write_message_result(results_dir, entry_number, record)
        summary["total"] += 1
        summary["passed" if evaluation["passed"] else "failed"] += 1
        if index < len(selected) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Forge sample evaluation set.")
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Delay between requests; defaults to five seconds.",
    )
    parser.add_argument("--start-entry", type=int, default=1)
    parser.add_argument("--end-entry", type=int, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Write each completed entry as messageN.json in this directory.",
    )
    args = parser.parse_args()
    summary = run(
        args.messages,
        args.labels,
        args.output,
        args.delay_seconds,
        args.start_entry,
        args.end_entry,
        args.results_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
