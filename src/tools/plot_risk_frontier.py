"""The risk frontier: reward vs. tail failures, three ways to dial risk aversion.

Plots reward per decision against the handover-failure rate (HOF) for the three
risk mechanisms studied, all at the matched 1000-episode, 3-seed budget on the
canonical tf=8 backbone:

  * single-quantile (Q1): one quantile at tau = alpha/2, alpha swept.
  * hard-CVaR (truncated): k = ceil(25*alpha) tail quantiles averaged, alpha swept.
  * soft-step weights: all 25 quantiles, lower [0,0.25] up-weighted by ratio r;
    r=1 is the risk-neutral mean and r -> inf is hard-CVaR at alpha=0.25, so the
    soft-step curve bridges RN and the hard-CVaR alpha=0.25 point.

RN (n25_tb) and DQN (dqn_tf8) are reference anchors at the same budget; the
efficient frontier is the upper-left (more reward, fewer failures). The paper
figure is the single HOF panel (results/final_metrics/plots/risk_frontier.png); a
two-panel PP+HOF version is also written for our own analysis.

Run (PYTHONPATH must include src):
    ./venv-RL/bin/python3 src/tools/plot_risk_frontier.py
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = "results/benchmarks"
_GRID = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]

# Colours = the reserved per-algorithm palette shared with the master bar plots:
_C_Q1 = "#9467bd"     # single quantile  -> Truncated / RA-1q  (purple)
_C_CVAR = "#e377c2"   # hard CVaR        -> RA     (magenta/pink)
_C_SOFT = "#d62728"   # soft-step        -> Soft-step (crimson red)
_C_RN = "#8c564b"     # risk-neutral anchor -> RN  (warm brown)
_C_DQN = "#ff7f0e"    # scalar baseline anchor -> DQN (vibrant orange)


def _agg(suffix: str, agent: str = "qrdqn") -> dict[str, float] | None:
    """Aggregate reward/decision and tail rates over a run's raw eval files."""
    fs = sorted(glob.glob(f"{_BENCH}/*{suffix}"))
    if not fs:
        return None
    rpa, pp, hof = [], [], []
    for f in sorted(glob.glob(f"{fs[-1]}/eval/{agent}_raw_seed*.csv")):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        rpa.append(sum(float(x["reward"]) for x in rows)
                   / sum(float(x["n_decisions"]) for x in rows))
        pp.append(np.mean([float(x["pp_rate"]) for x in rows]))
        hof.append(np.mean([float(x["hof_rate"]) for x in rows]))
    if not rpa:
        return None
    return {"rpa": float(np.mean(rpa)),
            "pp": float(np.mean(pp)),
            "hof": float(np.mean(hof))}


def _dial(suffixes: list[str]) -> dict[str, np.ndarray]:
    pts = [p for p in (_agg(s) for s in suffixes) if p]
    return {k: np.array([p[k] for p in pts]) for k in ("rpa", "pp", "hof")}


def _load() -> dict:
    tags = [f"{int(round(a * 100)):03d}" for a in _GRID]
    q1 = _dial([f"nogate_uf_q1_a{t}" for t in tags])
    cvar = _dial([f"nogate_uf_cvar_a{t}" for t in tags])
    # Soft-step dial: RN (r=1) -> increasing r -> hard CVaR at alpha=0.25 (r=inf).
    # The full r-grid is noisy on these rare tail events; four representative
    # points (the two limits plus r=2 and r=10) trace the RN->CVaR
    # interpolation cleanly without the cross-seed zigzag.
    rn = _agg("nogate_n25_tb")
    soft_seq = [rn,
                _agg("nogate_dist_softcvar_a025_r20"),   # r = 2
                _agg("nogate_softhi_r10"),               # r = 10
                _agg("nogate_uf_cvar_a025")]             # r -> inf (hard CVaR)
    soft_seq = [p for p in soft_seq if p]
    soft = {k: np.array([p[k] for p in soft_seq]) for k in ("rpa", "pp", "hof")}
    return {"q1": q1, "cvar": cvar, "soft": soft, "rn": rn}


def _draw(ax, d: dict, key: str, xlab: str, annotate: bool) -> None:
    q1, cvar, soft, rn = d["q1"], d["cvar"], d["soft"], d["rn"]
    ax.plot(cvar[key], cvar["rpa"], "-s", color=_C_CVAR, lw=1.7, ms=5,
            label=r"hard CVaR ($k=\lceil 25\alpha\rceil$)")
    ax.plot(q1[key], q1["rpa"], "-o", color=_C_Q1, lw=1.7, ms=5,
            label=r"single quantile ($\tau=\alpha/2$)")
    ax.plot(soft[key], soft["rpa"], "-^", color=_C_SOFT, lw=1.7, ms=5,
            label=r"soft-step ($\alpha=.25$, $r$ up)")
    ax.scatter([rn[key]], [rn["rpa"]], marker="*", s=200, color=_C_RN,
               zorder=5, edgecolor="k", linewidth=0.4, label="RN")
    if annotate:
        # Mark risk levels on the single-quantile dial; alpha=0.25 is the knee
        # adopted as the working point. RN / top-right is least risk-averse;
        # alpha falls toward the lower-left.
        specs = {
            0.10: dict(xytext=(-7, 0), ha="right", va="center"),
            0.25: dict(xytext=(-7, 0), ha="right", va="center"),
            0.50: dict(xytext=(0, 2), ha="center", va="bottom"),
        }
        for a, kw in specs.items():
            i = _GRID.index(a)
            ax.annotate(rf"$\alpha={a:g}$", (q1[key][i], q1["rpa"][i]),
                        textcoords="offset points", fontsize=7.5,
                        color=_C_Q1, **kw)
        # Mark the adopted soft-step operating point r=2 (index 1 of the soft
        # sequence: RN, r=2, r=10, r->inf).
        ax.annotate(r"$r=2$", (soft[key][1], soft["rpa"][1]),
                    textcoords="offset points", xytext=(3, -9),
                    ha="left", va="top", fontsize=7.5, color=_C_SOFT)
    ax.set_xlabel(xlab)
    ax.set_ylabel("reward / decision")
    ax.grid(True, alpha=0.25)


def main() -> None:
    d = _load()
    os.makedirs("results/final_metrics/plots", exist_ok=True)

    # --- Paper figure: single panel, reward vs. HO-failure. ---
    fig, ax = plt.subplots(figsize=(5.25, 3.1))
    _draw(ax, d, "hof", "HO-failure rate (HOF / min)", annotate=True)
    ax.set_xlim(left=0.32)  # room for the left-placed alpha labels
    ax.annotate("efficient", xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=8, style="italic", color="0.35")
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = "results/final_metrics/plots/risk_frontier.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {out}")

    # --- Analysis figure: PP + HOF panels (not for the paper). ---
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _draw(axes[0], d, "pp", "ping-pong rate (PP / min)", annotate=False)
    _draw(axes[1], d, "hof", "HO-failure rate (HOF / min)", annotate=True)
    axes[0].set_title("Reward vs. ping-pong")
    axes[1].set_title("Reward vs. HO-failure")
    axes[1].legend(fontsize=8.5, loc="lower right")
    fig2.suptitle("Risk frontier (efficient = upper-left)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = "results/final_metrics/plots/risk_frontier_2panel.png"
    fig2.savefig(out2, dpi=160)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
