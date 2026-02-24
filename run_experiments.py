#!/usr/bin/env python3
"""Strict 4-step ablation runner — run from inside NMAP_amin/ or from NCAP/."""

# gym_bridge must be the first real import to patch the gym namespace before
# any other module (tonic, environments) tries to import gym.
try:
    import NMAP_amin.gym_bridge  # noqa: F401  (when run from NCAP/)
except ModuleNotFoundError:
    import gym_bridge  # noqa: F401  (when run from inside NMAP_amin/)

import os
import shutil
import subprocess
from pathlib import Path


# ROOT is always NMAP_amin/ regardless of where the script lives.
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
TONIC_OUTPUT_DIRS = (
    ROOT / "outputs" / "training",
    ROOT / "outputs" / "training_logs",
    ROOT / "outputs" / "improved_mixed_env",
)

# ── Experiment definitions ────────────────────────────────────────────────────
# All commands use "main.py" (relative to ROOT / NMAP_amin/) and pass
# --training_steps 2000000 for the full 2M-step run.

EXPERIMENTS = [
    {
        "name": "01_baseline",
        "cmd": [
            "python",
            "main.py",
            "--mode", "train",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
        ],
    },
    {
        "name": "02_oscillation",
        "cmd": [
            "python",
            "main.py",
            "--mode", "train",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--force_oscillation",
        ],
    },
    {
        "name": "03_init",
        "cmd": [
            "python",
            "main.py",
            "--mode", "train",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--force_oscillation",
            "--sparse_init",
        ],
    },
    {
        "name": "04_full_nmap",
        "cmd": [
            "python",
            "main.py",
            "--mode", "train",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--force_oscillation",
            "--sparse_init",
            "--sparse_reg_lambda", "0.05",
        ],
    },
]


def _build_subprocess_env() -> dict[str, str]:
    """Build environment vars so subprocesses can import NMAP_amin and tonic."""
    env = os.environ.copy()
    cwd_abs = str(Path.cwd().resolve())
    pythonpath_parts = [
        cwd_abs,          # NMAP_amin/ (set by os.chdir below)
        str(ROOT),        # NMAP_amin/ explicitly
        str(ROOT.parent), # NCAP/ so 'import NMAP_amin.gym_bridge' also resolves
        str((ROOT / "tonic").resolve()),  # local tonic fork
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _archive_tonic_outputs(target_dir: Path) -> list[Path]:
    moved_paths: list[Path] = []
    for source_dir in TONIC_OUTPUT_DIRS:
        if not source_dir.exists():
            continue
        destination = target_dir / source_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source_dir), str(destination))
        moved_paths.append(destination)
    return moved_paths


def _clear_staging_outputs() -> None:
    for output_dir in TONIC_OUTPUT_DIRS:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def run_all_experiments() -> None:
    # Always execute from inside NMAP_amin/ so relative paths resolve correctly.
    os.chdir(ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env = _build_subprocess_env()

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        result_dir = RESULTS_ROOT / exp_name

        print("\n" + "=" * 100)
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"COMMAND: {' '.join(exp['cmd'])}")
        print("=" * 100)

        _clear_staging_outputs()
        subprocess.run(exp["cmd"], check=True, env=env)

        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        moved = _archive_tonic_outputs(result_dir)

        print("=" * 100)
        print(f"SUCCESS: {exp_name}")
        print(f"RESULT FOLDER: {result_dir}")
        if moved:
            print("MOVED OUTPUT DIRECTORIES:")
            for path in moved:
                print(f"  - {path}")
        else:
            print("MOVED OUTPUT DIRECTORIES: none found")
        print("=" * 100)


if __name__ == "__main__":
    run_all_experiments()
