"""Head-to-head comparison of two handover policies on ONE UE trajectory.

The UE path is precomputed, so it is identical for both policies; what differs
is the sequence of serving decisions and their consequences. This script runs
two policies (any of: LTM heuristic, DQN, RN, soft-step, RA-1q, RA) on the SAME
UE and renders one compact figure:

    +-----------+---------------------------------------------+
    |  context  |  R1  Serving SINR (dB)        A  vs  B       |
    |    map    |---------------------------------------------|  } tight pair
    | (UE path  |  R2  Serving spectral efficiency (bps/Hz)   |
    |  + HOs +  |---------------------------------------------|
    | time tags)|  R3  policy A  cell raster (per-sector)      |
    |-----------|---------------------------------------------|  } tight pair
    | scoreboard|  R4  policy B  cell raster (per-sector)      |
    +-----------+---------------------------------------------+

Everything on the right shares the time axis, so the two policies read as "two
lines, one per model" (R1/R2) and "two rasters, one per model" (R3/R4); the two
members of each pair sit close together since they share a plot style. Each BS
row in a raster is split into 3 thin sub-bands (its 3 sectors), so intra-BS
sector changes and inter-BS handovers are both visible. The scoreboard
summarises the per-UE KPIs (green = better).

Run from the repo root with PYTHONPATH exported (see CLAUDE.md):

    ./venv-RL/bin/python3 src/tools/plot_policy_comparison.py --a ltm --b rn --ue 900
    ./venv-RL/bin/python3 src/tools/plot_policy_comparison.py --a dqn --b rn --ue 900
    ./venv-RL/bin/python3 src/tools/plot_policy_comparison.py --a rn  --b ra --ue 900 --animate
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.agents.distributional.qrdqn import QRDQNAgent  # noqa: E402
from src.distrl.agents.standard.dqn import DQNAgent  # noqa: E402
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent  # noqa: E402
from src.distrl.envs.ltm_gym import LTMEnv  # noqa: E402
from src.distrl.utils.config import Config  # noqa: E402
from src.distrl.viz.dashboard import (  # noqa: E402
    DEFAULT_SECTOR_ANGLES_DEG,
    get_bs_sites,
)

DT = 0.01  # seconds per simulator tick (10 ms)
NUM_BS = 7
BS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]
A_COLOR = "#2c3e50"     # fallback policy-A colour (overridden per policy)
B_COLOR = "#e6194b"     # fallback policy-B colour (overridden per policy)
# One reserved colour per algorithm, so a policy looks the SAME across every
# comparison figure. Chosen to stay distinct from each other and from the
# failure-event colours below.
POLICY_COLORS = {
    "ltm": "#7f7f7f",       # grey (matching master bar plot)
    "dqn": "#ff7f0e",       # vibrant orange (matching master bar plot)
    "rn": "#8c564b",        # warm sienna brown (matching master bar plot)
    "softstep": "#d62728",  # crimson red (matching master bar plot)
    "ra1q": "#9467bd",      # royal purple (matching master bar plot)
    "ra": "#e377c2",        # soft pink / magenta (matching master bar plot)
}
# Failure-event colours: outside the BS palette AND the policy palette, so they
# never read as "a cell" or "a policy". Always drawn as dashed line + icon.
HOF_COLOR = "#00bcd4"   # cyan    — handover failure
RLF_COLOR = "#cc00cc"   # magenta — radio-link failure
OUTAGE_DB = 0.0         # SINR <= 0 dB starts the out-of-sync timer (RLF risk)
WIN_GREEN = "#d6f0d2"   # scoreboard "better value" highlight

_BMK = "results/benchmarks"
# name -> (agent_type, benchmark folder | None, model filename | None, label)
POLICIES: dict[str, tuple[str, str | None, str | None, str]] = {
    "ltm": ("ltm_baseline", None, None, "LTM baseline"),
    "dqn": ("dqn", "bmk_2026-06-04_10_nogate_final_dqn", "dqn_best.pth", "DQN"),
    "rn": ("qrdqn", "bmk_2026-06-04_11_nogate_final_rn", "qrdqn_best.pth", "RN"),
    "softstep": ("qrdqn", "bmk_2026-06-05_13_nogate_final_softcvar_r20",
                 "qrdqn_best.pth", "Soft-step"),
    "ra1q": ("qrdqn", "bmk_2026-06-04_14_nogate_final_ra_1q",
             "qrdqn_best.pth", "RA-1q"),
    "ra": ("qrdqn", "bmk_2026-06-04_12_nogate_final_ra",
           "qrdqn_best.pth", "RA (CVaR k=7)"),
}


def _resolve(name: str) -> tuple[str, str, str | None, str]:
    """Map a policy name to (agent_type, config_path, model_path, label)."""
    atype, folder, model, label = POLICIES[name]
    if folder is None:  # LTM baseline runs in its native gated config
        return atype, "configs/config.yaml", None, label
    return (atype, f"{_BMK}/{folder}/config.yaml",
            f"{_BMK}/{folder}/models/{model}", label)


def _build_agent(agent_type: str, env: LTMEnv):
    cfg = Config.get()["agent"]
    if agent_type == "dqn":
        return DQNAgent(cfg, env.observation_space, env.action_space, device="cpu")
    if agent_type == "qrdqn":
        return QRDQNAgent(cfg, env.observation_space, env.action_space, device="cpu")
    if agent_type == "ltm_baseline":
        return LTMBaselineAgent(
            cfg, env.observation_space, env.action_space, device="cpu"
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def run_policy(
    agent_type: str,
    ue_idx0: int,
    config_path: str,
    model_path: str | None,
    seed: int = 0,
) -> dict:
    """Run one policy on UE ``ue_idx0`` (0-based) and return per-tick traces."""
    Config.set_config_path(config_path)
    np.random.seed(seed)  # deterministic BLER / HOF / RLF draws across runs
    env = LTMEnv(config=Config.get())
    env.current_ue_idx = ue_idx0
    agent = _build_agent(agent_type, env)

    is_baseline = agent_type == "ltm_baseline"
    if not is_baseline:
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        agent.load(model_path)

    state, _ = env.reset()
    done = False
    while not done:
        if is_baseline:
            state, _, done, _, _ = env.step(0, high_res_callback=agent.select_action)
        else:
            action = agent.select_action(state, epsilon=0)
            state, _, done, _, _ = env.step(action)

    t = int(min(env.t, env.Max_iter))
    serving = np.asarray(env.ServingBSSector[:t], dtype=int)
    valid = serving >= 0
    serving_bs = np.where(valid, serving // 3, -1)

    sinr = np.full(t, np.nan)
    cols = np.arange(t)
    sinr[valid] = env.all_snir_episode[serving[valid], cols[valid]]

    prepared_sect = np.asarray(env.ReservedBSSectors[:, :t]) > 0  # (21, t)
    prepared_bs = prepared_sect.reshape(NUM_BS, 3, t).any(axis=1)  # (7, t)

    ho = np.asarray(env.HO_event[:t]) > 0
    pp = np.asarray(env.ping_pong[:t]) > 0
    hof = np.asarray(env.HOF[:t]) > 0
    rlf = np.asarray(env.RLF[:t]) > 0

    return {
        "T": t,
        "serving_bs": serving_bs,
        "serving_sect": serving,        # raw sector index per tick (neg = none)
        "sinr": sinr,
        "se": np.asarray(env.MCS[:t]),
        "prepared_bs": prepared_bs,
        "prepared_sect": prepared_sect,  # (21, t) per-sector prepared mask
        "ho": ho, "pp": pp, "hof": hof, "rlf": rlf,
        "ue_pos": np.asarray(env.ue_positions[:t]),
        "counts": {
            "HO": int(ho.sum()), "PP": int(pp.sum()),
            "HOF": int(hof.sum()), "RLF": int(rlf.sum()),
            "mean_sinr": float(np.nanmean(sinr)),
            "mean_se": float(np.nanmean(np.where(valid, env.MCS[:t], np.nan))),
            "prep_avg": float(prepared_bs.sum(axis=0).mean()),
        },
    }


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean that tolerates NaN gaps."""
    s = pd.Series(y).rolling(w, min_periods=max(1, w // 4), center=True).mean()
    out = s.to_numpy(copy=True)
    out[np.isnan(y)] = np.nan  # keep genuine "no serving cell" gaps as gaps
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encode a boolean array into (start, length) of True runs."""
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def _event_vlines(ax, res: dict) -> None:
    """Vertical dashed lines at this policy's HOF (cyan) and RLF (black) ticks."""
    for s in np.flatnonzero(res["hof"]):
        ax.axvline(s * DT, color=HOF_COLOR, ls="--", lw=0.9, alpha=0.55, zorder=3)
    for s in np.flatnonzero(res["rlf"]):
        ax.axvline(s * DT, color=RLF_COLOR, ls="--", lw=1.0, alpha=0.6, zorder=3)


def _draw_bs_layout(ax, fs: int = 10) -> np.ndarray:
    """Draw the 7 BS sites, hex cells, labels and per-BS sector divisions.

    Three short, semi-transparent lines (in the BS colour) mark the boundaries
    between a BS's 3 sectors, and a small S0/S1/S2 label sits in the middle of
    each sector wedge. The sector boresights (90/210/330 deg) were verified
    empirically: across ~280k served ticks the BS->UE bearing matches sector
    0/1/2 at 90/210/330 deg 98% of the time. Hence local sector ``s`` points at
    ``DEFAULT_SECTOR_ANGLES_DEG[s] + 60`` and the dividers fall on the
    ``DEFAULT_SECTOR_ANGLES_DEG`` (30/150/270 deg) boundaries. Returns the site
    coords.
    """
    sites = get_bs_sites()
    hex_side = 200.0 / np.sqrt(3)
    div_len = hex_side * 0.5                      # divider length (inside cell)
    lab_r = hex_side * 0.34                       # radius for the S0/S1/S2 labels
    div_ang = np.deg2rad(np.array(DEFAULT_SECTOR_ANGLES_DEG))         # boundaries
    lab_ang = np.deg2rad(np.array(DEFAULT_SECTOR_ANGLES_DEG) + 60.0)  # boresights
    for b, (x, y) in enumerate(sites):
        hx = hex_side * np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1]) + x
        hy = hex_side * np.sqrt(3) * np.array([0, -.5, -.5, 0, .5, .5, 0]) + y
        ax.plot(hx, hy, "--", color=BS_COLORS[b], lw=0.7, alpha=0.45, zorder=1)
        for ang in div_ang:
            ax.plot([x, x + div_len * np.cos(ang)],
                    [y, y + div_len * np.sin(ang)],
                    color=BS_COLORS[b], lw=1.0, alpha=0.35, zorder=1.5)
        for s, ang in enumerate(lab_ang):
            ax.text(x + lab_r * np.cos(ang), y + lab_r * np.sin(ang),
                    f"S{s}", fontsize=fs - 3.5, color=BS_COLORS[b],
                    alpha=0.55, ha="center", va="center", zorder=3)
        ax.scatter(x, y, marker="^", s=130, c=BS_COLORS[b],
                   edgecolors="k", linewidths=0.7, zorder=4)
        ax.text(x + 7, y + 20, f"BS{b}", fontsize=fs, fontweight="bold",
                color=BS_COLORS[b], zorder=5)
    return sites


def _time_markers(ax, path: np.ndarray, T: int, every: int = 60) -> None:
    """Small dots + labels every `every` seconds, linking the map to time."""
    for sec in range(every, int(T * DT), every):
        i = int(sec / DT)
        if i >= T:
            break
        ax.scatter(*path[i], marker="o", s=16, c="0.35", zorder=6)
        ax.annotate(f"{sec}s", path[i], fontsize=7.5, color="0.3",
                    xytext=(4, 4), textcoords="offset points", zorder=6)


def _draw_map(ax, a: dict, b: dict, ue: int, a_label: str, b_label: str,
              a_color: str, b_color: str) -> None:
    _draw_bs_layout(ax)
    path = a["ue_pos"]
    ax.plot(path[:, 0], path[:, 1], "-", color="0.6", lw=1.4, alpha=0.8,
            zorder=2, label="UE path")
    ax.scatter(*path[0], marker="o", s=95, c="#2ca02c", edgecolors="k",
               zorder=6, label="start")
    ax.scatter(*path[-1], marker="s", s=95, c="#444", edgecolors="k",
               zorder=6, label="end")
    _time_markers(ax, path, a["T"])

    for res, color, mk, lab in (
        (a, a_color, "o", f"{a_label} handover"),
        (b, b_color, "D", f"{b_label} handover"),
    ):
        idx = np.flatnonzero(res["ho"])
        if len(idx):
            pts = res["ue_pos"][idx]
            ax.scatter(pts[:, 0], pts[:, 1], marker=mk, s=48, c=color,
                       edgecolors="white", linewidths=0.6, zorder=7, label=lab)
    for res in (a, b):  # RLFs as black X on whichever policy
        idx = np.flatnonzero(res["rlf"])
        if len(idx):
            pts = res["ue_pos"][idx]
            ax.scatter(pts[:, 0], pts[:, 1], marker="x", s=95, c=RLF_COLOR,
                       linewidths=2.0, zorder=8)

    ax.set_title(f"UE {ue} trajectory  (handover locations + time tags)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)


def _short(label: str) -> str:
    """Compact tag for table headers, e.g. 'RA (CVaR k=7)' -> 'RA'."""
    return label.split(" (")[0].replace(" baseline", "").strip()


def _draw_scoreboard(ax, a: dict, b: dict, a_label: str, b_label: str,
                     a_color: str, b_color: str) -> None:
    """Compact KPI table; the better value in each row is highlighted green."""
    ax.axis("off")
    ca, cb = a["counts"], b["counts"]
    # (display name, key, direction)  direction: 'lo'=lower better, 'hi'=higher, ''=neutral
    rows = [
        ("Handovers / ep", "HO", "lo"),
        ("Ping-pongs", "PP", "lo"),
        ("HO failures (HOF)", "HOF", "lo"),
        ("Radio-link fails (RLF)", "RLF", "lo"),
        ("Mean SINR [dB]", "mean_sinr", "hi"),
        ("Mean SE [bps/Hz]", "mean_se", "hi"),
        ("Avg prepared cells", "prep_avg", ""),
    ]
    text, colors = [], []
    for name, key, d in rows:
        va, vb = ca[key], cb[key]
        sa = f"{va:.2f}" if isinstance(va, float) else str(va)
        sb = f"{vb:.2f}" if isinstance(vb, float) else str(vb)
        text.append([name, sa, sb])
        wa = wb = "white"
        if d and va != vb:
            a_better = (va < vb) if d == "lo" else (va > vb)
            wa, wb = (WIN_GREEN, "white") if a_better else ("white", WIN_GREEN)
        colors.append(["white", wa, wb])
    tbl = ax.table(cellText=text,
                   colLabels=["KPI (per UE)", _short(a_label), _short(b_label)],
                   cellColours=colors, colColours=["#ececec", "#f5f5f5", "#f5f5f5"],
                   cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl[(0, 1)].get_text().set_color(a_color)
    tbl[(0, 2)].get_text().set_color(b_color)
    for col in (1, 2):
        tbl[(0, col)].get_text().set_fontweight("bold")
    ax.set_title("Episode scoreboard  (green = better)", fontsize=10, pad=6,
                 fontweight="bold")


# Each BS row is split into its 3 sectors as thin, tightly-stacked sub-bands, so
# an intra-BS sector change reads as the solid band hopping among 3 close lines,
# while an inter-BS handover is the larger jump to a different BS row.
_ROW_BAND_H = 0.84        # full height of one BS row's band (old single-band height)
_SECT_GAP = 0.02          # tiny gap between a BS's 3 sector sub-bands


def _sector_band(row: int, s: int) -> tuple[float, float]:
    """(y0, height) for sector ``s`` (0..2) inside the band at y-row ``row``.

    Sectors stack top-to-bottom as S0, S1, S2 (so S0 is the top sub-band).
    """
    sub = _ROW_BAND_H / 3.0
    y0 = (row - _ROW_BAND_H / 2.0) + (2 - s) * sub + _SECT_GAP / 2.0
    return y0, sub - _SECT_GAP


def _draw_raster(ax, res: dict, color: str, label: str) -> None:
    t = res["T"]
    span = t * DT
    serving_sect = res["serving_sect"]
    prepared_sect = res["prepared_sect"]
    for b in range(NUM_BS):
        row = NUM_BS - 1 - b          # draw BS0 at the top, BS6 at the bottom
        for s in range(3):            # 3 sectors per BS, tightly stacked
            sect = b * 3 + s
            y0, h = _sector_band(row, s)
            # Prepared: light band. Serving: solid band on top.
            for st, ln in _runs(prepared_sect[sect]):
                ax.broken_barh([(st * DT, ln * DT)], (y0, h),
                               facecolors=BS_COLORS[b], alpha=0.22)
            for st, ln in _runs(serving_sect == sect):
                ax.broken_barh([(st * DT, ln * DT)], (y0, h),
                               facecolors=BS_COLORS[b], alpha=0.95)
    # HOF / RLF: vertical dashed lines spanning the raster + icon above the top row.
    _event_vlines(ax, res)
    for mask, mk, mc in ((res["hof"], "v", HOF_COLOR), (res["rlf"], "x", RLF_COLOR)):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        kw = (dict(facecolors="none", edgecolors=mc, linewidths=1.1)
              if mk == "v" else dict(c=mc))  # HOF triangles hollow, RLF solid
        ax.scatter(idx * DT, np.full(len(idx), NUM_BS - 0.3), marker=mk,
                   s=46, zorder=7, clip_on=False, **kw)
    ax.set_title(
        f"{label}  cells  (3 sectors / BS)",
        fontsize=10, color=color, fontweight="bold",
    )
    # Major ticks = BS rows (BS0 at top); minor ticks = the 3 sector sub-bands
    # (S0/S1/S2) within each row, so the y-axis shows both BS and sector. The
    # BS label is pushed further left (larger pad) so it sits beside, not on top
    # of, the centre sector label.
    ax.set_yticks(range(NUM_BS))
    ax.set_yticklabels([f"BS{NUM_BS - 1 - r}" for r in range(NUM_BS)], fontsize=7)
    sect_pos, sect_lab = [], []
    for r in range(NUM_BS):
        for s in range(3):
            y0, h = _sector_band(r, s)
            sect_pos.append(y0 + h / 2.0)
            sect_lab.append(f"S{s}")
    ax.set_yticks(sect_pos, minor=True)
    ax.set_yticklabels(sect_lab, minor=True)
    # The centre sub-band (S1) sits exactly on the BS major tick; keep it from
    # being auto-removed for coinciding with a major tick.
    ax.yaxis.remove_overlapping_locs = False
    ax.tick_params(axis="y", which="major", length=0, pad=19)
    ax.tick_params(axis="y", which="minor", length=0, pad=1,
                   labelsize=5.5, labelcolor="0.45")
    ax.set_ylim(-0.6, NUM_BS - 0.2)
    ax.set_xlim(0, span)
    ax.grid(True, axis="x", ls=":", alpha=0.3)


def _raster_legend(ax) -> None:
    """Raster encoding legend, placed on the RIGHT at the same height as the raster
    title (which is centered), so the two share one strip in the gap between the
    line-plot pair and the raster pair without occluding any bands."""
    leg = [
        mpatches.Patch(color="0.5", alpha=0.22, label="prepared"),
        mpatches.Patch(color="0.5", alpha=0.95, label="serving"),
        plt.Line2D([], [], marker="v", ls="--", color=HOF_COLOR,
                   markerfacecolor="none", label="HOF"),
        plt.Line2D([], [], marker="x", ls="--", color=RLF_COLOR, label="RLF"),
    ]
    ax.legend(handles=leg, loc="center right", bbox_to_anchor=(1.0, 1.085),
              fontsize=8, ncol=4, framealpha=0.9, handlelength=1.3,
              handletextpad=0.5, columnspacing=1.1, borderpad=0.4)


def _make_axes(fig):
    """Left column = map (top 3/4) + scoreboard (bottom 1/4); right column = two
    TIGHT pairs (SINR + SE, then raster A + raster B) with a wider gap between the
    pairs, since the two members of each pair share a plot style."""
    outer = fig.add_gridspec(1, 2, width_ratios=[1.55, 2.3], wspace=0.16,
                             top=0.935, bottom=0.045, left=0.035, right=0.995)
    left = outer[0, 0].subgridspec(4, 1, hspace=0.45)
    ax_map = fig.add_subplot(left[0:3, 0])
    ax_score = fig.add_subplot(left[3, 0])
    # Between-pair gap is just wide enough to hold the raster legend (placed
    # above raster A) plus its title; the rest of the height goes to the plots.
    # The raster pair gets more height than the line-plot pair (the rasters carry
    # 21 sector sub-bands and benefit from the room).
    right = outer[0, 1].subgridspec(2, 1, hspace=0.17, height_ratios=[1.0, 1.28])
    # hspace scales with axes height, and the raster pair is 1.28x taller, so the
    # top pair needs ~1.28x the hspace (0.17 vs 0.13) for an EQUAL physical gap.
    pair_top = right[0].subgridspec(2, 1, hspace=0.17)      # SINR + SE: tight
    pair_bot = right[1].subgridspec(2, 1, hspace=0.13)      # raster A + B: tight
    ax_sinr = fig.add_subplot(pair_top[0])
    ax_se = fig.add_subplot(pair_top[1], sharex=ax_sinr)
    ax_ra = fig.add_subplot(pair_bot[0], sharex=ax_sinr)
    ax_rb = fig.add_subplot(pair_bot[1], sharex=ax_sinr)
    return ax_map, ax_score, ax_sinr, ax_se, ax_ra, ax_rb


def build_figure(a: dict, b: dict, ue: int, a_label: str, b_label: str, out: str,
                 a_color: str = A_COLOR, b_color: str = B_COLOR) -> None:
    T = min(a["T"], b["T"])
    span = T * DT
    win = 60  # 0.6 s smoothing for the SINR / SE lines

    fig = plt.figure(figsize=(19, 11))
    ax_map, ax_score, ax_sinr, ax_se, ax_ra, ax_rb = _make_axes(fig)

    _draw_map(ax_map, a, b, ue, a_label, b_label, a_color, b_color)
    _draw_scoreboard(ax_score, a, b, a_label, b_label, a_color, b_color)

    # --- R1: serving SINR ---
    tsec = np.arange(T) * DT
    ax_sinr.axhspan(-30, OUTAGE_DB, color="red", alpha=0.06)
    ax_sinr.axhline(OUTAGE_DB, color="red", ls="--", lw=0.8, alpha=0.5)
    for res, color, lab in ((a, a_color, a_label), (b, b_color, b_label)):
        c = res["counts"]
        ax_sinr.plot(tsec, _smooth(res["sinr"][:T], win), color=color, lw=1.4,
                     alpha=0.7, label=f"{lab}  (mean {c['mean_sinr']:.1f} dB)")
        _event_vlines(ax_sinr, res)
        for mask, mk, mc in ((res["hof"][:T], "v", HOF_COLOR),
                             (res["rlf"][:T], "x", RLF_COLOR)):
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            kw = (dict(facecolors="none", edgecolors=mc, linewidths=1.2)
                  if mk == "v" else dict(c=mc))  # HOF triangles hollow, RLF solid
            ax_sinr.scatter(idx * DT, res["sinr"][idx], marker=mk, s=55,
                            zorder=6, **kw)
    ax_sinr.set_ylabel("Serving SINR [dB]")
    ax_sinr.set_title("Serving-cell SINR  (cyan ▽ = HOF, magenta ✗ = RLF, "
                      "shaded = OOS zone)",
                      fontsize=10, fontweight="bold")
    ax_sinr.legend(loc="upper right", fontsize=8, ncol=2)
    ax_sinr.grid(True, ls=":", alpha=0.4)

    # --- R2: serving spectral efficiency ---
    for res, color, lab in ((a, a_color, a_label), (b, b_color, b_label)):
        c = res["counts"]
        ax_se.plot(tsec, _smooth(res["se"][:T], win), color=color, lw=1.4,
                   alpha=0.7, label=f"{lab}  (mean {c['mean_se']:.2f} bps/Hz)")
        _event_vlines(ax_se, res)
    ax_se.set_ylabel("Spectral eff. [bps/Hz]")
    ax_se.set_title("Serving-cell spectral efficiency (throughput proxy)",
                    fontsize=10, fontweight="bold")
    ax_se.legend(loc="lower right", fontsize=8, ncol=2)
    ax_se.grid(True, ls=":", alpha=0.4)

    # --- R3 / R4: cell rasters ---
    _draw_raster(ax_ra, a, a_color, a_label)
    _draw_raster(ax_rb, b, b_color, b_label)
    ax_rb.set_xlabel("time [s]")
    _raster_legend(ax_ra)  # above raster A, filling the inter-pair gap

    for ax in (ax_sinr, ax_se, ax_ra):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_sinr.set_xlim(0, span)
    ax_sinr.set_xticks(np.arange(0, span, 60))  # 60 s ticks, matching the map tags

    fig.suptitle(
        f"{a_label}  vs.  {b_label}   —   same UE {ue}, same trajectory, "
        f"different handover decisions",
        fontsize=14, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def animate_figure(a: dict, b: dict, ue: int, a_label: str, b_label: str, out: str,
                   a_color: str = A_COLOR, b_color: str = B_COLOR,
                   fps: int = 16, target_frames: int = 320) -> None:
    """Animated comparison: a time cursor sweeps the episode.

    Lines (SINR/SE) and the map reveal progressively; the cell rasters are drawn
    in full and a receding "future curtain" uncovers them as time passes.
    """
    import matplotlib.animation as animation

    T = min(a["T"], b["T"])
    span = T * DT
    win = 60
    tsec = np.arange(T) * DT
    sm = {
        "as": _smooth(a["sinr"][:T], win), "bs": _smooth(b["sinr"][:T], win),
        "ae": _smooth(a["se"][:T], win), "be": _smooth(b["se"][:T], win),
    }

    fig = plt.figure(figsize=(19, 11))
    ax_map, ax_score, ax_sinr, ax_se, ax_ra, ax_rb = _make_axes(fig)
    _draw_scoreboard(ax_score, a, b, a_label, b_label, a_color, b_color)

    # --- map: static layout + full faint path + time tags, dynamic UE/links/HO ---
    sites = _draw_bs_layout(ax_map)
    path = a["ue_pos"]
    ax_map.plot(path[:, 0], path[:, 1], "-", color="0.83", lw=1.0, zorder=1)
    ax_map.scatter(*path[0], marker="o", s=95, c="#2ca02c", edgecolors="k", zorder=5)
    _time_markers(ax_map, path, T)
    ax_map.set_title(f"UE {ue} trajectory", fontsize=12, fontweight="bold")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal")
    ax_map.grid(True, ls=":", alpha=0.4)
    traj_done, = ax_map.plot([], [], "-", color="0.45", lw=1.8, zorder=2)
    # Serving links: semi-transparent + different widths so the overlap (both
    # policies on the same BS) reads as B-on-A rather than a single line.
    link_a, = ax_map.plot([], [], "-", color=a_color, lw=3.0, alpha=0.55, zorder=6)
    link_b, = ax_map.plot([], [], "-", color=b_color, lw=1.7, alpha=0.65, zorder=7)
    ue_dot, = ax_map.plot([], [], "o", ms=11, color="black", zorder=8)
    ho_a = ax_map.scatter([], [], marker="o", s=48, c=a_color, edgecolors="white",
                          linewidths=0.6, zorder=7, label=f"{a_label} handover")
    ho_b = ax_map.scatter([], [], marker="D", s=48, c=b_color, edgecolors="white",
                          linewidths=0.6, zorder=7, label=f"{b_label} handover")
    ax_map.legend(loc="upper right", fontsize=8.5)
    ha_t = np.flatnonzero(a["ho"][:T]); ha_p = a["ue_pos"][ha_t]
    hb_t = np.flatnonzero(b["ho"][:T]); hb_p = b["ue_pos"][hb_t]

    # --- SINR / SE: static decor + event lines, dynamic lines + cursor ---
    ax_sinr.axhspan(-30, OUTAGE_DB, color="red", alpha=0.06)
    ax_sinr.axhline(OUTAGE_DB, color="red", ls="--", lw=0.8, alpha=0.5)
    for res in (a, b):
        _event_vlines(ax_sinr, res)
        _event_vlines(ax_se, res)
    sinr_a, = ax_sinr.plot([], [], color=a_color, lw=1.4, alpha=0.7,
                           label=f"{a_label} (mean {a['counts']['mean_sinr']:.1f} dB)")
    sinr_b, = ax_sinr.plot([], [], color=b_color, lw=1.4, alpha=0.7,
                           label=f"{b_label} (mean {b['counts']['mean_sinr']:.1f} dB)")
    se_a, = ax_se.plot([], [], color=a_color, lw=1.4, alpha=0.7,
                       label=f"{a_label} (mean {a['counts']['mean_se']:.2f})")
    se_b, = ax_se.plot([], [], color=b_color, lw=1.4, alpha=0.7,
                       label=f"{b_label} (mean {b['counts']['mean_se']:.2f})")
    sall = np.concatenate([sm["as"], sm["bs"]])
    ax_sinr.set_ylim(min(np.nanmin(sall) - 3, -2), np.nanmax(sall) + 3)
    eall = np.concatenate([sm["ae"], sm["be"]])
    ax_se.set_ylim(0, np.nanmax(eall) + 0.5)
    ax_sinr.set_ylabel("Serving SINR [dB]")
    ax_se.set_ylabel("Spectral eff. [bps/Hz]")
    ax_sinr.set_title("Serving-cell SINR  (cyan ▽ = HOF, magenta ✗ = RLF, "
                      "shaded = OOS zone)", fontsize=10, fontweight="bold")
    ax_se.set_title("Serving-cell spectral efficiency (throughput proxy)",
                    fontsize=10, fontweight="bold")
    ax_sinr.legend(loc="upper right", fontsize=8, ncol=2)
    ax_se.legend(loc="lower right", fontsize=8, ncol=2)
    for ax in (ax_sinr, ax_se):
        ax.grid(True, ls=":", alpha=0.4)

    # --- rasters: full draw + receding curtain + cursor ---
    _draw_raster(ax_ra, a, a_color, a_label)
    _draw_raster(ax_rb, b, b_color, b_label)
    ax_rb.set_xlabel("time [s]")
    _raster_legend(ax_ra)  # above raster A, filling the inter-pair gap
    curtains = []
    for ax in (ax_ra, ax_rb):
        curt = mpatches.Rectangle((0, -1), span, NUM_BS + 2, color="white",
                                  alpha=0.82, zorder=6)
        ax.add_patch(curt)
        curtains.append(curt)
    cursors = [ax.axvline(0, color="0.25", lw=1.1, alpha=0.8, zorder=8)
               for ax in (ax_sinr, ax_se, ax_ra, ax_rb)]

    for ax in (ax_sinr, ax_se, ax_ra):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_sinr.set_xlim(0, span)
    ax_sinr.set_xticks(np.arange(0, span, 60))  # 60 s ticks, matching the map tags
    fig.suptitle(f"{a_label}  vs.  {b_label}   —   UE {ue}, same trajectory",
                 fontsize=14, fontweight="bold")

    def animate(tf: int):
        cx = tf * DT
        traj_done.set_data(path[:tf + 1, 0], path[:tf + 1, 1])
        ue_dot.set_data([path[tf, 0]], [path[tf, 1]])
        for link, res in ((link_a, a), (link_b, b)):
            bs = res["serving_bs"][tf]
            if bs >= 0:
                link.set_data([path[tf, 0], sites[bs, 0]],
                              [path[tf, 1], sites[bs, 1]])
            else:
                link.set_data([], [])
        ho_a.set_offsets(ha_p[ha_t <= tf] if (ha_t <= tf).any() else np.empty((0, 2)))
        ho_b.set_offsets(hb_p[hb_t <= tf] if (hb_t <= tf).any() else np.empty((0, 2)))
        sinr_a.set_data(tsec[:tf + 1], sm["as"][:tf + 1])
        sinr_b.set_data(tsec[:tf + 1], sm["bs"][:tf + 1])
        se_a.set_data(tsec[:tf + 1], sm["ae"][:tf + 1])
        se_b.set_data(tsec[:tf + 1], sm["be"][:tf + 1])
        for curt in curtains:
            curt.set_x(cx)
            curt.set_width(max(span - cx, 1e-6))
        for cur in cursors:
            cur.set_xdata([cx, cx])
        return [traj_done, ue_dot, link_a, link_b, ho_a, ho_b,
                sinr_a, sinr_b, se_a, se_b, *curtains, *cursors]

    step = max(1, T // target_frames)
    frames = list(range(0, T, step)) + [T - 1]
    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2600)
    ani.save(out, writer=writer)
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Single-policy view (one algorithm by itself, e.g. to introduce the LTM
# baseline / the environment). Same panels as the comparison, minus the second
# policy: map + scoreboard on the left; SINR, SE and ONE raster on the right.
# The per-plot proportions match the comparison figure (line plots : raster =
# 1 : 1.28), but every panel is larger because there are 3 right-column plots
# instead of 4.
# --------------------------------------------------------------------------- #
_SINR_TITLE = ("Serving-cell SINR  (cyan ▽ = HOF, magenta ✗ = RLF, "
               "shaded = OOS zone)")
_SE_TITLE = "Serving-cell spectral efficiency (throughput proxy)"


def _make_axes_single(fig):
    """Left column = map (top 3/4) + scoreboard (bottom 1/4); right column =
    SINR + SE (tight pair) then ONE raster, with the same per-plot size ratio as
    the comparison figure (line plot : raster = 1 : 1.28)."""
    outer = fig.add_gridspec(1, 2, width_ratios=[1.55, 2.3], wspace=0.16,
                             top=0.935, bottom=0.045, left=0.035, right=0.995)
    left = outer[0, 0].subgridspec(4, 1, hspace=0.45)
    ax_map = fig.add_subplot(left[0:3, 0])
    ax_score = fig.add_subplot(left[3, 0])
    # Line pair (two plots, weight 2.0 total -> 1.0 each) over a single raster
    # (weight 1.28) keeps the comparison figure's per-plot ratio. The gap before
    # the raster holds the legend; the line pair shares a tight sub-grid.
    right = outer[0, 1].subgridspec(2, 1, hspace=0.17, height_ratios=[2.0, 1.28])
    pair_top = right[0].subgridspec(2, 1, hspace=0.17)      # SINR + SE: tight
    ax_sinr = fig.add_subplot(pair_top[0])
    ax_se = fig.add_subplot(pair_top[1], sharex=ax_sinr)
    ax_raster = fig.add_subplot(right[1], sharex=ax_sinr)
    return ax_map, ax_score, ax_sinr, ax_se, ax_raster


def _draw_map_single(ax, res: dict, ue: int, label: str, color: str) -> None:
    _draw_bs_layout(ax)
    path = res["ue_pos"]
    ax.plot(path[:, 0], path[:, 1], "-", color="0.6", lw=1.4, alpha=0.8,
            zorder=2, label="UE path")
    ax.scatter(*path[0], marker="o", s=95, c="#2ca02c", edgecolors="k",
               zorder=6, label="start")
    ax.scatter(*path[-1], marker="s", s=95, c="#444", edgecolors="k",
               zorder=6, label="end")
    _time_markers(ax, path, res["T"])
    idx = np.flatnonzero(res["ho"])
    if len(idx):
        pts = res["ue_pos"][idx]
        ax.scatter(pts[:, 0], pts[:, 1], marker="o", s=48, c=color,
                   edgecolors="white", linewidths=0.6, zorder=7,
                   label=f"{label} handover")
    idx = np.flatnonzero(res["rlf"])
    if len(idx):
        pts = res["ue_pos"][idx]
        ax.scatter(pts[:, 0], pts[:, 1], marker="x", s=95, c=RLF_COLOR,
                   linewidths=2.0, zorder=8, label="RLF")
    ax.set_title(f"UE {ue} trajectory  (handover locations + time tags)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)


def _draw_scoreboard_single(ax, res: dict, label: str, color: str) -> None:
    """Single-policy KPI table (no comparison, so no green highlighting)."""
    ax.axis("off")
    c = res["counts"]
    rows = [
        ("Handovers / ep", "HO"),
        ("Ping-pongs", "PP"),
        ("HO failures (HOF)", "HOF"),
        ("Radio-link fails (RLF)", "RLF"),
        ("Mean SINR [dB]", "mean_sinr"),
        ("Mean SE [bps/Hz]", "mean_se"),
        ("Avg prepared cells", "prep_avg"),
    ]
    text = []
    for name, key in rows:
        v = c[key]
        text.append([name, f"{v:.2f}" if isinstance(v, float) else str(v)])
    tbl = ax.table(cellText=text, colLabels=["KPI (per UE)", _short(label)],
                   colColours=["#ececec", "#f5f5f5"], cellLoc="center",
                   bbox=[0.0, 0.0, 1.0, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl[(0, 1)].get_text().set_color(color)
    tbl[(0, 1)].get_text().set_fontweight("bold")
    ax.set_title("Episode scoreboard", fontsize=10, pad=6, fontweight="bold")


def build_figure_single(res: dict, ue: int, label: str, out: str,
                        color: str = A_COLOR) -> None:
    T = res["T"]
    span = T * DT
    win = 60

    fig = plt.figure(figsize=(19, 11))
    ax_map, ax_score, ax_sinr, ax_se, ax_raster = _make_axes_single(fig)
    _draw_map_single(ax_map, res, ue, label, color)
    _draw_scoreboard_single(ax_score, res, label, color)

    tsec = np.arange(T) * DT
    c = res["counts"]

    # --- SINR ---
    ax_sinr.axhspan(-30, OUTAGE_DB, color="red", alpha=0.06)
    ax_sinr.axhline(OUTAGE_DB, color="red", ls="--", lw=0.8, alpha=0.5)
    ax_sinr.plot(tsec, _smooth(res["sinr"][:T], win), color=color, lw=1.5,
                 alpha=0.85, label=f"{label}  (mean {c['mean_sinr']:.1f} dB)")
    _event_vlines(ax_sinr, res)
    for mask, mk, mc in ((res["hof"][:T], "v", HOF_COLOR),
                         (res["rlf"][:T], "x", RLF_COLOR)):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        kw = (dict(facecolors="none", edgecolors=mc, linewidths=1.2)
              if mk == "v" else dict(c=mc))  # HOF triangles hollow, RLF solid
        ax_sinr.scatter(idx * DT, res["sinr"][idx], marker=mk, s=55, zorder=6, **kw)
    ax_sinr.set_ylabel("Serving SINR [dB]")
    ax_sinr.set_title(_SINR_TITLE, fontsize=10, fontweight="bold")
    ax_sinr.legend(loc="upper right", fontsize=8)
    ax_sinr.grid(True, ls=":", alpha=0.4)

    # --- SE ---
    ax_se.plot(tsec, _smooth(res["se"][:T], win), color=color, lw=1.5,
               alpha=0.85, label=f"{label}  (mean {c['mean_se']:.2f} bps/Hz)")
    _event_vlines(ax_se, res)
    ax_se.set_ylabel("Spectral eff. [bps/Hz]")
    ax_se.set_title(_SE_TITLE, fontsize=10, fontweight="bold")
    ax_se.legend(loc="lower right", fontsize=8)
    ax_se.grid(True, ls=":", alpha=0.4)

    # --- raster ---
    _draw_raster(ax_raster, res, color, label)
    ax_raster.set_xlabel("time [s]")
    _raster_legend(ax_raster)

    for ax in (ax_sinr, ax_se):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_sinr.set_xlim(0, span)
    ax_sinr.set_xticks(np.arange(0, span, 60))  # 60 s ticks, matching the map tags

    fig.suptitle(
        f"{label}   —   UE {ue}: trajectory, serving quality, and cell decisions",
        fontsize=14, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def animate_figure_single(res: dict, ue: int, label: str, out: str,
                          color: str = A_COLOR, fps: int = 16,
                          target_frames: int = 320) -> None:
    """Animated single-policy view: a time cursor sweeps the episode; lines and
    the map reveal progressively, the raster is uncovered by a receding curtain."""
    import matplotlib.animation as animation

    T = res["T"]
    span = T * DT
    win = 60
    tsec = np.arange(T) * DT
    sm = {"s": _smooth(res["sinr"][:T], win), "e": _smooth(res["se"][:T], win)}

    fig = plt.figure(figsize=(19, 11))
    ax_map, ax_score, ax_sinr, ax_se, ax_raster = _make_axes_single(fig)
    _draw_scoreboard_single(ax_score, res, label, color)

    # --- map: static layout + faint path + tags, dynamic UE/link/HO ---
    sites = _draw_bs_layout(ax_map)
    path = res["ue_pos"]
    ax_map.plot(path[:, 0], path[:, 1], "-", color="0.83", lw=1.0, zorder=1)
    ax_map.scatter(*path[0], marker="o", s=95, c="#2ca02c", edgecolors="k", zorder=5)
    _time_markers(ax_map, path, T)
    ax_map.set_title(f"UE {ue} trajectory", fontsize=12, fontweight="bold")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal")
    ax_map.grid(True, ls=":", alpha=0.4)
    traj_done, = ax_map.plot([], [], "-", color="0.45", lw=1.8, zorder=2)
    link, = ax_map.plot([], [], "-", color=color, lw=2.4, alpha=0.7, zorder=6)
    ue_dot, = ax_map.plot([], [], "o", ms=11, color="black", zorder=8)
    ho_m = ax_map.scatter([], [], marker="o", s=48, c=color, edgecolors="white",
                          linewidths=0.6, zorder=7, label=f"{label} handover")
    ax_map.legend(loc="upper right", fontsize=8.5)
    h_t = np.flatnonzero(res["ho"][:T]); h_p = res["ue_pos"][h_t]

    # --- SINR / SE: static decor + event lines, dynamic line + cursor ---
    ax_sinr.axhspan(-30, OUTAGE_DB, color="red", alpha=0.06)
    ax_sinr.axhline(OUTAGE_DB, color="red", ls="--", lw=0.8, alpha=0.5)
    _event_vlines(ax_sinr, res)
    _event_vlines(ax_se, res)
    sinr_l, = ax_sinr.plot([], [], color=color, lw=1.5, alpha=0.85,
                           label=f"{label} (mean {res['counts']['mean_sinr']:.1f} dB)")
    se_l, = ax_se.plot([], [], color=color, lw=1.5, alpha=0.85,
                       label=f"{label} (mean {res['counts']['mean_se']:.2f})")
    ax_sinr.set_ylim(min(np.nanmin(sm["s"]) - 3, -2), np.nanmax(sm["s"]) + 3)
    ax_se.set_ylim(0, np.nanmax(sm["e"]) + 0.5)
    ax_sinr.set_ylabel("Serving SINR [dB]")
    ax_se.set_ylabel("Spectral eff. [bps/Hz]")
    ax_sinr.set_title(_SINR_TITLE, fontsize=10, fontweight="bold")
    ax_se.set_title(_SE_TITLE, fontsize=10, fontweight="bold")
    ax_sinr.legend(loc="upper right", fontsize=8)
    ax_se.legend(loc="lower right", fontsize=8)
    for ax in (ax_sinr, ax_se):
        ax.grid(True, ls=":", alpha=0.4)

    # --- raster: full draw + receding curtain + cursor ---
    _draw_raster(ax_raster, res, color, label)
    ax_raster.set_xlabel("time [s]")
    _raster_legend(ax_raster)
    curt = mpatches.Rectangle((0, -1), span, NUM_BS + 2, color="white",
                              alpha=0.82, zorder=6)
    ax_raster.add_patch(curt)
    cursors = [ax.axvline(0, color="0.25", lw=1.1, alpha=0.8, zorder=8)
               for ax in (ax_sinr, ax_se, ax_raster)]
    for ax in (ax_sinr, ax_se):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_sinr.set_xlim(0, span)
    ax_sinr.set_xticks(np.arange(0, span, 60))
    fig.suptitle(f"{label}   —   UE {ue}", fontsize=14, fontweight="bold")

    def animate(tf: int):
        cx = tf * DT
        traj_done.set_data(path[:tf + 1, 0], path[:tf + 1, 1])
        ue_dot.set_data([path[tf, 0]], [path[tf, 1]])
        bs = res["serving_bs"][tf]
        if bs >= 0:
            link.set_data([path[tf, 0], sites[bs, 0]], [path[tf, 1], sites[bs, 1]])
        else:
            link.set_data([], [])
        ho_m.set_offsets(h_p[h_t <= tf] if (h_t <= tf).any() else np.empty((0, 2)))
        sinr_l.set_data(tsec[:tf + 1], sm["s"][:tf + 1])
        se_l.set_data(tsec[:tf + 1], sm["e"][:tf + 1])
        curt.set_x(cx)
        curt.set_width(max(span - cx, 1e-6))
        for cur in cursors:
            cur.set_xdata([cx, cx])
        return [traj_done, ue_dot, link, ho_m, sinr_l, se_l, curt, *cursors]

    step = max(1, T // target_frames)
    frames = list(range(0, T, step)) + [T - 1]
    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2600)
    ani.save(out, writer=writer)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="ltm", choices=list(POLICIES),
                   help="first policy (drawn dark)")
    p.add_argument("--b", default="rn", choices=list(POLICIES),
                   help="second policy (drawn red); ignored with --single")
    p.add_argument("--single", action="store_true",
                   help="plot only policy --a by itself (no comparison)")
    p.add_argument("--ue", type=int, default=900, help="1-based UE number")
    p.add_argument("--animate", action="store_true",
                   help="render an MP4 with a sweeping time cursor instead of a PNG")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    ue0 = args.ue - 1
    ext = "mp4" if args.animate else "png"

    if args.single:
        atype, cfg, model, label = _resolve(args.a)
        print(f"Running {label} on UE {args.ue} ...")
        res = run_policy(atype, ue0, cfg, model)
        print(f"  {res['counts']}")
        color = POLICY_COLORS.get(args.a, A_COLOR)
        out = args.out or (f"figures/10_single_policy/{args.a}/"
                           f"{args.a}_ue{args.ue}.{ext}")
        fn = animate_figure_single if args.animate else build_figure_single
        fn(res, args.ue, label, out, color)
        return

    results = {}
    for tag in (args.a, args.b):
        atype, cfg, model, label = _resolve(tag)
        print(f"Running {label} on UE {args.ue} ...")
        results[tag] = (run_policy(atype, ue0, cfg, model), label)
        print(f"  {results[tag][0]['counts']}")
    (a_res, a_label), (b_res, b_label) = results[args.a], results[args.b]

    a_color = POLICY_COLORS.get(args.a, A_COLOR)
    b_color = POLICY_COLORS.get(args.b, B_COLOR)
    pair = f"{args.a}_vs_{args.b}"
    out = args.out or f"figures/09_head_to_head/{pair}/{pair}_ue{args.ue}.{ext}"
    fn = animate_figure if args.animate else build_figure
    fn(a_res, b_res, args.ue, a_label, b_label, out, a_color, b_color)


if __name__ == "__main__":
    main()
