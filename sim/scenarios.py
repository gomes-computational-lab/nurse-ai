from __future__ import annotations

import json
from pathlib import Path

from sim.models import Scenario


SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def list_scenarios() -> list[Scenario]:
    return [load_scenario(path.stem) for path in sorted(SCENARIO_DIR.glob("*.json"))]


def load_scenario(scenario_id: str) -> Scenario:
    path = SCENARIO_DIR / f"{scenario_id}.json"
    if not path.exists():
        available = ", ".join(s.id for s in list_scenarios()) or "none"
        raise FileNotFoundError(f"Unknown scenario '{scenario_id}'. Available scenarios: {available}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return Scenario(**data)

