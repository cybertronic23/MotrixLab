# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Run separate A/B processes with identical arguments and restore go1.py."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    cfg = ROOT / "motrix_envs/src/motrix_envs/locomotion/quadruped/go1.py"
    original = cfg.read_text()
    before = "ControlConfig(action_scale=0.1)"
    after = "ControlConfig(action_scale=0.05)"
    if original.count(before) != 1:
        raise ValueError("Expected exactly one Go1-specific default action_scale=0.1")
    output = ROOT / "runs/week05-evidence"
    command = [
        sys.executable,
        str(ROOT / "experiments/week05-go1/record_follow.py"),
        "--policy",
        str(Path(args.policy).resolve()),
        "--seconds",
        "10",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    print("Both processes use:", " ".join(command), flush=True)
    try:
        for group in ("A", "B"):
            cfg.write_text(original if group == "A" else original.replace(before, after))
            with (output / f"replay-{group}.log").open("w") as stream:
                subprocess.run(
                    command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300
                )
            print(f"{group} completed", flush=True)
    finally:
        cfg.write_text(original)
        print("Restored original Go1 configuration", flush=True)


if __name__ == "__main__":
    main()
