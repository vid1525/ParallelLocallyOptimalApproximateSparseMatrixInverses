from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
SCRIPTS_DIR = ROOT / "scripts"

CASE_OPTIONS = {
    "dataset": "--datasets",
    "num_iterations": "--max-iterations",
    "max_density": "--max-density",
    "num_tries": "--tries",
    "num_threads": "--num-threads",
}


def load_scenarios(config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f'{config_path} must contain a "scenarios" list')
    return scenarios


def build_command(script_type: str, case: dict[str, Any]) -> list[str]:
    if not script_type or Path(script_type).name != script_type:
        raise ValueError(f"Invalid script_type: {script_type!r}")

    script_path = SCRIPTS_DIR / f"{script_type}.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Scenario script does not exist: {script_path}")

    command = [sys.executable, str(script_path)]
    supported_fields = {"name", "dropping", *CASE_OPTIONS}
    unsupported_fields = set(case).difference(supported_fields)
    if unsupported_fields:
        fields = ", ".join(sorted(unsupported_fields))
        raise ValueError(f"Unsupported case field(s): {fields}")

    for field, option in CASE_OPTIONS.items():
        if field in case:
            value = case[field]
            if field == "num_threads" and value == "max":
                value = os.cpu_count() or 1
            command.extend((option, str(value)))

    dropping = case.get("dropping", True)
    if not isinstance(dropping, bool):
        raise ValueError('Case field "dropping" must be a boolean')
    if not dropping:
        command.append("--disable-dropping")

    return command


def main() -> None:
    scenarios = load_scenarios()

    for scenario in scenarios:
        scenario_name = scenario.get("name", "Unnamed scenario")
        script_type = scenario.get("script_type")
        cases = scenario.get("cases")

        if not isinstance(script_type, str):
            raise ValueError(f'Scenario "{scenario_name}" has no valid script_type')
        if not isinstance(cases, list):
            raise ValueError(f'Scenario "{scenario_name}" has no valid cases list')

        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f'Scenario "{scenario_name}" contains an invalid case')

            case_name = case.get("name", "Unnamed case")
            case_options = dict(case)
            if "num_threads" in scenario:
                case_options.setdefault("num_threads", scenario["num_threads"])
            command = build_command(script_type, case_options)
            print(f"Running: {scenario_name} / {case_name}", flush=True)
            print(f"Command: {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
