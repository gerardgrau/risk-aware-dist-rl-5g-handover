"""Generate the single-policy presentation set (one algorithm by itself).

Renders each requested policy ON ITS OWN — map + scoreboard on the left, serving
SINR / SE / per-sector cell raster on the right — as both a static PNG and an
animated MP4, for the standard presentation UEs. Useful for introducing the
environment / the LTM baseline before the head-to-head comparisons.

    ./venv-RL/bin/python3 src/tools/gen_single_policy_set.py
    ./venv-RL/bin/python3 src/tools/gen_single_policy_set.py --policies ltm rn
    ./venv-RL/bin/python3 src/tools/gen_single_policy_set.py --no-video
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.plot_policy_comparison import (  # noqa: E402
    POLICIES, POLICY_COLORS, A_COLOR, _resolve, animate_figure_single,
    build_figure_single, run_policy,
)

OUT = "figures/10_single_policy"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["ltm"],
                   choices=list(POLICIES), help="policies to render by themselves")
    p.add_argument("--ues", nargs="+", type=int, default=[250, 650, 900],
                   help="1-based UE numbers")
    p.add_argument("--no-video", action="store_true",
                   help="render only the static PNGs (skip the MP4s)")
    args = p.parse_args()

    for pol in args.policies:
        atype, cfg, model, label = _resolve(pol)
        color = POLICY_COLORS.get(pol, A_COLOR)
        for ue in args.ues:
            print(f"\n===== {label}: UE {ue} =====")
            res = run_policy(atype, ue - 1, cfg, model)
            stem = f"{OUT}/{pol}/{pol}_ue{ue}"   # per-policy subfolder
            build_figure_single(res, ue, label, f"{stem}.png", color)
            if not args.no_video:
                animate_figure_single(res, ue, label, f"{stem}.mp4", color)


if __name__ == "__main__":
    main()
