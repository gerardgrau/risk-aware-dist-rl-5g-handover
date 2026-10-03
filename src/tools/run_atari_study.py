"""Sweep QR-DQN quantile-positioning modes on Atari.

For each variant in ``configs/atari/*.yaml`` (or a user-selected subset),
trains for ``--frames`` frames on ``--game`` and records the final greedy
evaluation return. Prints a comparison table at the end.

The default frame budget (500k) is enough for the dynamics of each
variant to be distinguishable; serious Atari numbers need ~10M frames.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import yaml

DEFAULT_VARIANTS = [
    "qrdqn_midpoint",
    "qrdqn_gauss_legendre",
    "qrdqn_trapezoidal",
    "qrdqn_cvar_full",
    "qrdqn_cvar_truncated",
]

CFG_DIR = "configs/atari"


def detect_agent_type(cfg_path: str) -> str:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return str(cfg.get("agent", {}).get("type", "qrdqn"))


def find_latest_run(game: str, description: str) -> str | None:
    pattern = os.path.join("results", "atari", f"{game}_{description}_*")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_final_eval(run_dir: str) -> float | None:
    path = os.path.join(run_dir, "final_eval.txt")
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("mean_return="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def run_one(
    variant: str, game: str, frames: int, device: str, seed: int, threads: int,
) -> tuple[float, str | None, float | None]:
    cfg = os.path.join(CFG_DIR, f"{variant}.yaml")
    if not os.path.exists(cfg):
        raise FileNotFoundError(cfg)
    agent_type = detect_agent_type(cfg)
    cmd = [
        "venv-RL/bin/python3", "src/atari_main.py",
        "--config", cfg,
        "--game", game,
        "--frames", str(frames),
        "--device", device,
        "--seed", str(seed),
        "--description", variant,
        "--agent", agent_type,
    ]
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    print(f"\n>>> {variant} on {game}")
    print("    " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    run_dir = find_latest_run(game, variant)
    return elapsed, run_dir, read_final_eval(run_dir) if run_dir else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--frames", type=int, default=500_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of variant names.")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="Cap PyTorch intra-op threads (passed through to atari_main).",
    )
    args = parser.parse_args()

    variants = DEFAULT_VARIANTS if args.only is None else [
        v.strip() for v in args.only.split(",") if v.strip()
    ]

    rows: list[tuple[str, float, str | None, float | None]] = []
    for variant in variants:
        try:
            elapsed, run_dir, final_eval = run_one(
                variant, args.game, args.frames, args.device, args.seed,
                args.threads,
            )
        except subprocess.CalledProcessError as err:
            print(f"!!! {variant} failed: {err}", file=sys.stderr)
            rows.append((variant, float("nan"), None, None))
            continue
        rows.append((variant, elapsed, run_dir, final_eval))
        print(f"--- {variant} done in {elapsed/60:.1f}min, eval = {final_eval}")

    print("\n" + "=" * 78)
    print(f"ATARI {args.game.upper()} SWEEP SUMMARY  (frames={args.frames})")
    print("=" * 78)
    print(f"{'variant':<28}{'time (min)':>12}{'final_eval':>14}")
    print("-" * 78)
    for name, dur, _, fe in rows:
        fe_str = f"{fe:.2f}" if fe is not None else "nan"
        print(f"{name:<28}{dur/60:>12.1f}{fe_str:>14}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
