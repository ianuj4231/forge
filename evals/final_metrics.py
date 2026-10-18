from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "evals" / "results"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "final_metrics.json"


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "actual" in value and "expected" in value:
        return [value]
    return []


def collect_records(paths: list[Path]) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidates.extend(path.rglob("*.json"))
            candidates.extend(path.rglob("*.jsonl"))
        elif path.is_file():
            candidates.append(path)

    latest_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for path in candidates:
        for record in load_records(path):
            message_id = record.get("id") or record.get("actual", {}).get("id")
            if not message_id:
                continue
            timestamp = path.stat().st_mtime
            current = latest_by_id.get(message_id)
            if current is None or timestamp >= current[0]:
                latest_by_id[message_id] = (timestamp, record)
    return [item[1] for item in sorted(latest_by_id.values(), key=lambda item: item[1].get("id", ""))]


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    urgency_correct = 0
    action_exact = 0
    action_recall_values: list[float] = []
    action_precision_values: list[float] = []
    missed_emergencies: list[dict[str, Any]] = []
    must_not_violations: list[dict[str, Any]] = []

    for record in records:
        expected = record.get("expected", {})
        actual = record.get("actual", {})
        message_id = record.get("id", actual.get("id"))
        expected_actions = set(expected.get("expected_actions", []))
        actual_actions = set(actual.get("actions", []))
        accepted_urgency = set(expected.get("accepted_urgency", []))
        forbidden_actions = set(expected.get("must_not", []))
        violations = sorted(forbidden_actions & actual_actions)

        urgency_ok = not accepted_urgency or actual.get("urgency") in accepted_urgency
        urgency_correct += int(urgency_ok)
        action_exact += int(expected_actions == actual_actions)
        action_recall_values.append(len(expected_actions & actual_actions) / len(expected_actions) if expected_actions else 1.0)
        action_precision_values.append(len(expected_actions & actual_actions) / len(actual_actions) if actual_actions else (1.0 if not expected_actions else 0.0))

        emergency_expected = "escalate_emergency" in expected_actions
        if emergency_expected and "escalate_emergency" not in actual_actions:
            missed_emergencies.append({"id": message_id, "expected_actions": sorted(expected_actions), "actual_actions": sorted(actual_actions)})
        if violations:
            must_not_violations.append({"id": message_id, "must_not": sorted(forbidden_actions), "violations": violations, "actual_actions": sorted(actual_actions)})

    emergency_cases = sum("escalate_emergency" in record.get("expected", {}).get("expected_actions", []) for record in records)
    emergency_recall = (emergency_cases - len(missed_emergencies)) / emergency_cases if emergency_cases else 1.0
    return {
        "total_cases": total,
        "urgency_accuracy": urgency_correct / total if total else 0.0,
        "action_exact_accuracy": action_exact / total if total else 0.0,
        "action_recall": sum(action_recall_values) / total if total else 0.0,
        "action_precision": sum(action_precision_values) / total if total else 0.0,
        "emergency_cases": emergency_cases,
        "emergency_recall": emergency_recall,
        "missed_emergencies": missed_emergencies,
        "must_not_violations": must_not_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate final evaluation metrics from saved result files.")
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_RESULTS])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = calculate_metrics(collect_records(args.paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
