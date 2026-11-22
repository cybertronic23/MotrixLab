# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Record one fresh Go1 replay and its measured state, without altering the policy.

Run the identical command in two fresh processes, changing only the Go1
action_scale in go1.py between runs. The evidence directory is selected from
the actual environment configuration, rather than a command-line label.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from motrix_rl_rslrl.torch.wrap_np import RslrlNpEnvWrap

from motrix_env_motrixsim.renderer import MotrixSimRenderer
from motrix_rl.config import PlayConfig

ROOT = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(ROOT))
    from scripts import play

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    policy = Path(args.policy).resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    rows = []
    measured = {}
    original_init = RslrlNpEnvWrap.__init__
    original_step = RslrlNpEnvWrap.step
    original_capture = MotrixSimRenderer.capture

    def measured_init(self, env, device, render=None):
        original_init(self, env, device, render)
        measured.update(
            action_scale=float(env.cfg.control_config.action_scale),
            ctrl_dt=float(env.cfg.ctrl_dt),
            policy=str(policy),
            checkpoint_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
            seed=42,
            num_envs=1,
            device=str(device),
            observation_space=str(env.observation_space),
            action_space=str(env.action_space),
        )
        print("ACTUAL_REPLAY_CONFIG " + json.dumps(measured), flush=True)

    def measured_step(self, actions):
        result = original_step(self, actions)
        env = self._env
        state = self._state
        row = {
            "step": len(rows) + 1,
            "local_velocity": env.get_local_linvel()[0].tolist(),
            "upvector": env.get_gravity()[0].tolist(),
            "base_position": np.asarray(env.sim_data["base_pos"][0]).reshape(-1).tolist(),
            "joint_position": env.get_dof_pos()[0].tolist(),
            "default_angles": env.default_angles.tolist(),
            "actions": actions[0].detach().cpu().numpy().tolist(),
            "feet_contact": env.feet_contact[0].tolist(),
            "reward": float(state.reward[0]),
            "terminated": bool(state.terminated[0]),
            "truncated": bool(state.truncated[0]),
        }
        rows.append(row)
        if len(rows) == 1:
            print("FIRST_REPLAY_STEP " + json.dumps(row), flush=True)
        return result

    def follow_capture(self):
        position = self._data_source().dof_pos[0, :3]
        self._render.system_camera.set_view([float(position[0]), float(position[1]), 0.35], 2.8, -15.0, 135.0)
        return original_capture(self)

    RslrlNpEnvWrap.__init__ = measured_init
    RslrlNpEnvWrap.step = measured_step
    MotrixSimRenderer.capture = follow_capture
    try:
        play.run(
            PlayConfig(
                env="go1-walk-flat",
                policy=str(policy),
                num_envs=1,
                seed=42,
                record_video=True,
                record_seconds=args.seconds,
                record_width=960,
                record_height=540,
            )
        )
    finally:
        RslrlNpEnvWrap.__init__ = original_init
        RslrlNpEnvWrap.step = original_step
        MotrixSimRenderer.capture = original_capture
    if not rows:
        raise RuntimeError("No replay steps were recorded")
    group = "A" if measured["action_scale"] == 0.1 else "B"
    output = ROOT / "runs/week05-evidence" / group
    output.mkdir(parents=True, exist_ok=True)
    video = policy.parent.parent / "play_video.mp4"
    shutil.copy2(video, output / "replay.mp4")
    velocity = np.array([r["local_velocity"] for r in rows])
    height = np.array([r["base_position"][2] for r in rows])
    measured.update(
        recorded_video_seconds=args.seconds,
        steps=len(rows),
        simulated_seconds=len(rows) * measured["ctrl_dt"],
        mean_forward_velocity=float(velocity[:, 0].mean()),
        std_forward_velocity=float(velocity[:, 0].std()),
        mean_lateral_velocity=float(velocity[:, 1].mean()),
        min_base_height=float(height.min()),
        max_base_height=float(height.max()),
        terminations=sum(r["terminated"] for r in rows),
        truncations=sum(r["truncated"] for r in rows),
    )
    (output / "summary.json").write_text(json.dumps(measured, indent=2))
    with (output / "trajectory.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print("MEASURED_REPLAY_SUMMARY " + json.dumps(measured), flush=True)
    print(f"Saved evidence: {output}", flush=True)


if __name__ == "__main__":
    main()
