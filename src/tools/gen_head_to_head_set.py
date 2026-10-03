"""Generate the standard presentation set of head-to-head comparisons.

For each UE it runs the needed policies once, then renders the four
"progression" comparisons (LTM->DQN, DQN->RN, RN->RA, LTM->RN) as both a static
PNG and an animated MP4. Policy colours are the reserved per-algorithm palette,
so a given policy looks identical across every figure.

    ./venv-RL/bin/python3 src/tools/gen_head_to_head_set.py
    ./venv-RL/bin/python3 src/tools/gen_head_to_head_set.py --no-video
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.plot_policy_comparison import (  # noqa: E402
    POLICY_COLORS, _resolve, animate_figure, build_figure, run_policy,
)

UES = [250, 650, 900]
PAIRS = [("ltm", "dqn"), ("dqn", "rn"), ("rn", "ra"), ("ltm", "rn")]
OUT = "figures/09_head_to_head"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-video", action="store_true",
                   help="render only the static PNGs (skip the MP4s)")
    args = p.parse_args()

    needed = sorted({t for pair in PAIRS for t in pair})
    for ue in UES:
        print(f"\n===== UE {ue}: running {needed} =====")
        res: dict[str, tuple[dict, str]] = {}
        for tag in needed:
            atype, cfg, model, label = _resolve(tag)
            res[tag] = (run_policy(atype, ue - 1, cfg, model), label)
        for a, b in PAIRS:
            (ar, al), (br, bl) = res[a], res[b]
            stem = f"{OUT}/{a}_vs_{b}/{a}_vs_{b}_ue{ue}"  # per-comparison subfolder
            build_figure(ar, br, ue, al, bl, f"{stem}.png",
                         POLICY_COLORS[a], POLICY_COLORS[b])
            if not args.no_video:
                animate_figure(ar, br, ue, al, bl, f"{stem}.mp4",
                               POLICY_COLORS[a], POLICY_COLORS[b])


if __name__ == "__main__":
    main()
