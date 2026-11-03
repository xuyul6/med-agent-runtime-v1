"""Synthetic hospital catalog: departments and bookable slots. No real institution is represented."""
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

Window = Literal["am", "pm"]


@dataclass(frozen=True)
class Department:
    id: str
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Hospital:
    id: str
    name: str
    departments: tuple[Department, ...]

    @property
    def short_name(self) -> str:
        return self.name.replace("（模拟）", "")


@dataclass(frozen=True)
class Slot:
    slot_id: str
    hospital_id: str
    hospital: str
    department_id: str
    department: str
    day: date
    window: Window
    capacity: int


@dataclass(frozen=True)
class HospitalCatalog:
    hospitals: tuple[Hospital, ...]
    capacity: dict[str, int]
    horizon_days: int

    @classmethod
    def load(cls, path: Path) -> "HospitalCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hospitals = tuple(Hospital(item["id"], item["name"], tuple(Department(dept["id"], dept["name"], tuple(dept["aliases"]))
                                                                   for dept in item["departments"])) for item in data["hospitals"])
        return cls(hospitals, dict(data["capacity"]), int(data["horizon_days"]))

    def match_department(self, text: str, hospital_id: str | None = None) -> list[tuple[Hospital, Department]]:
        return [(hospital, dept) for hospital in self.hospitals if hospital_id in (None, hospital.id)
                for dept in hospital.departments if any(name in (text or "") for name in (dept.name, *dept.aliases))]

    def match_hospital(self, text: str) -> list[Hospital]:
        text = (text or "").strip()
        if not text:
            return []
        return [hospital for hospital in self.hospitals if hospital.short_name in text or text in hospital.short_name
                or hospital.short_name.removeprefix("演示") in text]

    def match_department_id(self, hospital_id: str, department_id: str) -> bool:
        return self._department(hospital_id, department_id) is not None

    def bookable(self, day: date, today: date) -> bool:
        return today < day <= today + timedelta(days=self.horizon_days)

    def _capacity(self, day: date, window: Window) -> int:
        return self.capacity.get(("weekend" if day.weekday() >= 5 else "weekday") + "_" + window, 0)

    def _department(self, hospital_id: str, department_id: str) -> tuple[Hospital, Department] | None:
        return next(((hospital, dept) for hospital in self.hospitals if hospital.id == hospital_id
                     for dept in hospital.departments if dept.id == department_id), None)

    def slots(self, hospital_id: str, department_id: str, day: date, window: Window | None, today: date) -> list[Slot]:
        found = self._department(hospital_id, department_id)
        if found is None or not self.bookable(day, today):
            return []
        hospital, dept = found
        return [Slot(f"{hospital.id}:{dept.id}:{day.isoformat()}:{part}", hospital.id, hospital.name, dept.id, dept.name, day, part, size)
                for part in ([window] if window else ["am", "pm"]) if (size := self._capacity(day, part)) > 0]

    def slot(self, slot_id: str) -> Slot | None:
        parts = (slot_id or "").split(":")
        if len(parts) != 4 or parts[3] not in {"am", "pm"}:
            return None
        try:
            day = date.fromisoformat(parts[2])
        except ValueError:
            return None
        found = self._department(parts[0], parts[1])
        if found is None or (size := self._capacity(day, parts[3])) <= 0:
            return None
        hospital, dept = found
        return Slot(slot_id, hospital.id, hospital.name, dept.id, dept.name, day, parts[3], size)

    def department_names(self) -> list[str]:
        return list(dict.fromkeys(dept.name for hospital in self.hospitals for dept in hospital.departments))
