from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ClinicData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.clinic = self._load("clinic.json")
        self.patients = self._load("patients.json")
        self.medications = self._load("medications.json")
        self.appointments = self._load("appointments.json")

    def _load(self, name: str) -> Any:
        return json.loads((self.data_dir / name).read_text(encoding="utf-8-sig"))

    def find_patient(self, name: str, contact: str) -> tuple[dict[str, Any] | None, str]:
        normalized_name = normalize_text(name)
        normalized_contact = normalize_contact(contact)
        for patient in self.patients:
            if normalized_contact and normalized_contact in {
                normalize_contact(patient.get("phone", "")),
                normalize_contact(patient.get("email", "")),
            }:
                return patient, "patient_contact"
            caregiver = patient.get("authorised_caregiver") or {}
            if normalized_contact and normalized_contact in {
                normalize_contact(caregiver.get("phone", "")),
                normalize_contact(caregiver.get("email", "")),
            }:
                return patient, "caregiver_contact"
        for patient in self.patients:
            if normalized_name and normalized_name == normalize_text(patient.get("name", "")):
                return patient, "patient_name_only"
            caregiver = patient.get("authorised_caregiver") or {}
            if normalized_name and normalized_name == normalize_text(caregiver.get("name", "")):
                return patient, "caregiver_name_only"
        return None, "not_found"

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def normalize_contact(value: str) -> str:
    value = value.strip().casefold()
    if "@" in value:
        return value
    return re.sub(r"\D", "", value)


def normalize_medication(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().casefold())
    return value.replace("–", "-")


def exact_medication_match(
    items: list[dict[str, Any]],
    requested: str | None,
) -> dict[str, Any] | None:
    if not requested:
        return None
    requested_normalized = normalize_medication(requested)
    matches = []
    for item in items:
        medication = normalize_medication(str(item.get("medication", "")))
        generic = re.split(r"\s+(?=\d|hfa|pen|inhaler)", medication, maxsplit=1)[0]
        if medication == requested_normalized or generic == requested_normalized:
            matches.append(item)
    return matches[0] if len(matches) == 1 else None
