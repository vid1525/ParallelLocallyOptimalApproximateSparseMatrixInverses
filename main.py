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
    "backward_error_realizations": "--backward-error-realizations",
    "max_backward_error_iterations": "--max-backward-error-iterations",
}


RESULTS_SUBDIRECTORIES = {
    "methods_convergence": "results_sample",
    "methods_large_convergence": "results_large",
    "methods_thread_speedup": "results_parallelization_speedup",
}
LARGE_DATASETS_SUBDIRECTORY = "datasets"


def load_scenarios(config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f'{config_path} must contain a "scenarios" list')
    return scenarios


def build_command(
    script_type: str,
    case: dict[str, Any],
    results_dir: Path | None = None,
    write_preconditioners: bool = False,
) -> list[str]:
    if not script_type or Path(script_type).name != script_type:
        raise ValueError(f"Invalid script_type: {script_type!r}")

    script_path = SCRIPTS_DIR / f"{script_type}.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Scenario script does not exist: {script_path}")

    command = [sys.executable, str(script_path)]
    supported_fields = {"name", "enabled", "dropping", "backward_error", *CASE_OPTIONS}
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

    if "backward_error" in case:
        backward_error = case["backward_error"]
        if not isinstance(backward_error, bool):
            raise ValueError('Case field "backward_error" must be a boolean')
        if script_type in {"methods_convergence", "methods_large_convergence"}:
            command.append("--backward-error" if backward_error else "--no-backward-error")
        elif backward_error:
            raise ValueError(f'{script_type} does not support backward-error calculation')

    if results_dir is not None:
        if script_type == "fetch_large_spd_matrices":
            command.extend(("--output-dir", str(results_dir / LARGE_DATASETS_SUBDIRECTORY)))
        else:
            output_subdirectory = RESULTS_SUBDIRECTORIES.get(script_type)
            if output_subdirectory is not None:
                command.extend(("--output-dir", str(results_dir / output_subdirectory)))
            if script_type == "methods_large_convergence":
                command.extend(("--datasets-dir", str(results_dir / LARGE_DATASETS_SUBDIRECTORY)))

    if write_preconditioners and script_type in {"methods_convergence", "methods_large_convergence"}:
        command.append("--write-preconditioners")

    return command


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the experiment scenarios declared in config.json."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Writable root directory for experiment outputs and downloaded large datasets.",
    )
    parser.add_argument(
        "--write-preconditioners",
        action="store_true",
        help="Write generated preconditioner .mtx files (disabled by default).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve() if args.results_dir is not None else None
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

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
            enabled = case_options.pop("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(
                    f'Case "{case_name}" in scenario "{scenario_name}" has a non-boolean enabled value'
                )
            if not enabled:
                print(f"Skipping: {scenario_name} / {case_name} (disabled)", flush=True)
                continue
            for field in ("num_threads", "backward_error", "backward_error_realizations"):
                if field in scenario:
                    case_options.setdefault(field, scenario[field])
            command = build_command(
                script_type,
                case_options,
                results_dir=results_dir,
                write_preconditioners=args.write_preconditioners,
            )
            print(f"Running: {scenario_name} / {case_name}", flush=True)
            print(f"Command: {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
