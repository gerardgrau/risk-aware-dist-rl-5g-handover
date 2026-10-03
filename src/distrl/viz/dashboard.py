import os
from typing import Any, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# BS LAYOUT CONSTANTS
# ------------------------------------------------------------
# Derived from the MATLAB reference. The 7-site hexagonal
# layout sits inside a 600 m x 600 m area; adjacent BS sites
# are spaced 200 m apart (InterBSHSpacing). Sector orientation
# vectors follow the standard 3GPP tri-sector convention
# (boresights at 30 deg, 150 deg, 270 deg), and the antenna
# pattern shape matches the MATLAB `r_theta` definition:
#     r_theta = -min(12*(theta/PatternWidth)^2, 30) * exp(i*theta)
# rotated to the sector boresight and scaled by 3 for drawing.
# ============================================================
DEFAULT_INTER_BS_SPACING = 200.0
DEFAULT_PATTERN_WIDTH = np.deg2rad(65.0)
DEFAULT_SECTOR_ANGLES_DEG = (30.0, 150.0, 270.0)
SECTOR_COLORS = ('#D95319', '#EDB120', '#77AC30')


def _default_sector_orient_vectors(num_bs: int) -> np.ndarray:
    angles = np.deg2rad(np.array(DEFAULT_SECTOR_ANGLES_DEG))
    per_bs = np.exp(1j * angles)
    return np.tile(per_bs, num_bs)


class MobilityDashboard:
    """Visualization dashboard for the LTM Handover environment.

    Renders three side-by-side panels:
      * Mobility map with hexagonal cells, sector boresights and antenna
        patterns (matches the MATLAB reference plot).
      * Real-time RSRP trace per BS (strongest sector).
      * Real-time MCS trace per BS — the spectral efficiency the UE
        would get if connected, i.e. *why* the agent picks one BS over
        another (higher MCS → higher reward).
    """

    def __init__(
        self,
        bs_sites: np.ndarray,
        sector_orient_vectors: np.ndarray | None = None,
        inter_bs_spacing: float = DEFAULT_INTER_BS_SPACING,
        pattern_width: float = DEFAULT_PATTERN_WIDTH,
    ) -> None:
        """
        Args:
            bs_sites: (num_bs, 2) array of BS site centre coordinates.
            sector_orient_vectors: (num_bs * 3,) complex unit vectors,
                one per global sector (BS_idx * 3 + local_sector). If
                None, falls back to the 3GPP 30 deg / 150 deg / 270 deg
                convention shared across all BS.
            inter_bs_spacing: Distance between adjacent BS sites (m).
                Used to size the hex cell (side = spacing / sqrt(3)),
                boresight indicator and antenna pattern.
            pattern_width: Antenna 3 dB beamwidth in radians.
        """
        self.bs_sites = bs_sites
        self.num_bs = bs_sites.shape[0]
        self.num_sectors = self.num_bs * 3

        if sector_orient_vectors is None:
            sector_orient_vectors = _default_sector_orient_vectors(self.num_bs)
        self.sector_orient_vectors = sector_orient_vectors
        self.inter_bs_spacing = inter_bs_spacing
        self.pattern_width = pattern_width

        self.bs_colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
        ]

    # ------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------
    def _draw_bs_layout(self, ax) -> None:
        hex_side = self.inter_bs_spacing / np.sqrt(3)

        theta = np.arange(-np.pi, np.pi + 0.01, 0.01)
        f_theta = np.minimum(12.0 * (theta / self.pattern_width) ** 2, 30.0)
        r_theta = -f_theta * np.exp(1j * theta)

        for bs_idx in range(self.num_bs):
            x_bs, y_bs = self.bs_sites[bs_idx]

            ax.scatter(
                x_bs, y_bs,
                c=self.bs_colors[bs_idx], marker='^', s=90,
                edgecolors='k', linewidths=0.6, alpha=0.95, zorder=4,
                label="BS" if bs_idx == 0 else None,
            )
            ax.text(
                x_bs + self.inter_bs_spacing / 30.0,
                y_bs + self.inter_bs_spacing / 10.0,
                f"BS {bs_idx}",
                fontsize=10, fontweight='bold', color=self.bs_colors[bs_idx],
                zorder=5,
            )

            hx = hex_side * np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1]) + x_bs
            hy = (hex_side * np.sqrt(3)
                  * np.array([0, -0.5, -0.5, 0, 0.5, 0.5, 0])) + y_bs
            ax.plot(hx, hy, 'k--', linewidth=0.5, alpha=0.4, zorder=1)

            for s_idx in range(3):
                global_sect = bs_idx * 3 + s_idx
                angle = np.angle(self.sector_orient_vectors[global_sect])

                line_len = self.inter_bs_spacing / 5.0
                ax.plot(
                    [x_bs, x_bs + line_len * np.cos(angle)],
                    [y_bs, y_bs + line_len * np.sin(angle)],
                    color='k', linewidth=0.8, alpha=0.5, zorder=2,
                )

                pattern = 3.0 * r_theta * np.exp(1j * angle)
                ax.plot(
                    pattern.real + x_bs, pattern.imag + y_bs,
                    color=SECTOR_COLORS[s_idx], linewidth=1.0, alpha=0.35,
                    zorder=2,
                    label=f"Sector {s_idx + 1}" if bs_idx == 0 else None,
                )

    def _setup_metric_axis(
        self,
        ax,
        title: str,
        ylabel: str,
        ylim: tuple[float, float],
    ) -> tuple[list, list]:
        ax.set_title(title)
        ax.set_xlabel("RL step (100 ms)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

        bg_lines, active_lines = [], []
        for i in range(self.num_bs):
            l_bg, = ax.plot(
                [], [], color=self.bs_colors[i], linewidth=1.0, alpha=0.25,
            )
            l_act, = ax.plot(
                [], [], color=self.bs_colors[i], linewidth=1.8, alpha=1.0,
                label=f"BS {i}",
            )
            bg_lines.append(l_bg)
            active_lines.append(l_act)
        ax.legend(loc='upper right', fontsize='small', ncol=4)
        ax.grid(True, linestyle=':', alpha=0.5)
        return bg_lines, active_lines

    def _preprocess_per_bs(
        self,
        metric_2d: np.ndarray,
        serving_sector_history: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns max-per-BS time series and serving mask."""
        total_steps = metric_2d.shape[0]
        bs_max = np.zeros((total_steps, self.num_bs))
        serving_mask = np.zeros((total_steps, self.num_bs), dtype=bool)

        for t in range(total_steps):
            curr_sect = serving_sector_history[t]
            curr_bs = curr_sect // 3 if curr_sect != -1 else -1

            for b in range(self.num_bs):
                sects = metric_2d[t, b * 3:b * 3 + 3]
                bs_max[t, b] = sects.max()
                if curr_bs == b:
                    serving_mask[t, b] = True

        return bs_max, serving_mask

    # ------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------
    def render_episode(
        self,
        history: dict[str, list[Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """Render the animation.

        Args:
            history: Dict with keys:
                'ue_pos'      -> list[T] of (x, y)
                'serving_bs'  -> list[T] of int (global sector index, -1 if none)
                'rsrp'        -> list[T] of (num_sectors,) RSRP per sector
                'mcs'         -> list[T] of (num_sectors,) MCS per sector
            save_path: If ends with '.mp4', writes via FFMpeg. Otherwise
                uses Pillow GIF. If None, shows the figure interactively.
        """
        ue_positions = np.array(history['ue_pos'])
        serving_sector_history = np.array(history['serving_bs'])
        rsrp_history = np.array(history['rsrp'])
        mcs_history = np.array(history['mcs'])

        total_steps = len(ue_positions)

        fig = plt.figure(figsize=(18, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_rsrp = fig.add_subplot(gs[0, 1])
        ax_mcs = fig.add_subplot(gs[1, 1])

        # --- Map setup ---
        ax_map.set_title("LTM-HO Mobility Map")
        self._draw_bs_layout(ax_map)

        ue_dot, = ax_map.plot([], [], 'ko', markersize=7, label="UE", zorder=10)
        conn_line, = ax_map.plot(
            [], [], 'k-', linewidth=1.5, alpha=0.7, label="Serving Link", zorder=5,
        )
        traj_line, = ax_map.plot([], [], 'k-', alpha=0.2, linewidth=1, zorder=3)

        margin = self.inter_bs_spacing * 0.6
        ax_map.set_xlim(
            np.min(self.bs_sites[:, 0]) - margin,
            np.max(self.bs_sites[:, 0]) + margin,
        )
        ax_map.set_ylim(
            np.min(self.bs_sites[:, 1]) - margin,
            np.max(self.bs_sites[:, 1]) + margin,
        )
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        ax_map.grid(True, linestyle=':', alpha=0.4)
        ax_map.set_aspect('equal')
        ax_map.legend(loc='upper right', fontsize='small')

        # --- RSRP panel ---
        rsrp_bg, rsrp_act = self._setup_metric_axis(
            ax_rsrp,
            title="Real-time RSRP (Strongest Sector per BS)",
            ylabel="RSRP (dBm)",
            ylim=(-115, -40),
        )
        rsrp_bs_max, rsrp_mask = self._preprocess_per_bs(
            rsrp_history, serving_sector_history,
        )

        # --- MCS panel ---
        mcs_bg, mcs_act = self._setup_metric_axis(
            ax_mcs,
            title="Real-time MCS (Strongest Sector per BS)",
            ylabel="Spectral Efficiency [bps/Hz]",
            ylim=(0, 7),
        )
        mcs_bs_max, mcs_mask = self._preprocess_per_bs(
            mcs_history, serving_sector_history,
        )

        # --- HO event markers (any serving-sector change, intra- or cross-BS) ---
        ho_label_done = False
        for t_step in range(1, len(serving_sector_history)):
            curr = serving_sector_history[t_step]
            prev = serving_sector_history[t_step - 1]
            if curr < 0 or prev < 0 or curr == prev:
                continue
            label = "Handover" if not ho_label_done else None
            ho_label_done = True
            for ax in (ax_rsrp, ax_mcs):
                ax.axvline(
                    t_step, color='#c62828', linestyle='--',
                    linewidth=1.0, alpha=0.6,
                    label=label if ax is ax_rsrp else None,
                )
        # Refresh legend on RSRP so the new "Handover" entry is included
        ax_rsrp.legend(loc='upper right', fontsize='small', ncol=4)

        def animate(t):
            ue_dot.set_data([ue_positions[t, 0]], [ue_positions[t, 1]])
            traj_line.set_data(ue_positions[:t + 1, 0], ue_positions[:t + 1, 1])

            curr_sect = serving_sector_history[t]
            if curr_sect != -1 and curr_sect < self.num_sectors:
                bs_idx = curr_sect // 3
                conn_line.set_color(self.bs_colors[bs_idx])
                conn_line.set_data(
                    [ue_positions[t, 0], self.bs_sites[bs_idx, 0]],
                    [ue_positions[t, 1], self.bs_sites[bs_idx, 1]],
                )
            else:
                conn_line.set_data([], [])

            x_vals = np.arange(t + 1)
            artists = [ue_dot, conn_line, traj_line]

            for (bs_max, mask, bg_lines, act_lines) in (
                (rsrp_bs_max, rsrp_mask, rsrp_bg, rsrp_act),
                (mcs_bs_max, mcs_mask, mcs_bg, mcs_act),
            ):
                for i in range(self.num_bs):
                    bg_lines[i].set_data(x_vals, bs_max[:t + 1, i])
                    active_y = bs_max[:t + 1, i].copy()
                    active_y[~mask[:t + 1, i]] = np.nan
                    act_lines[i].set_data(x_vals, active_y)
                artists += bg_lines + act_lines

            ax_rsrp.set_xlim(t - 140, t + 10)
            ax_mcs.set_xlim(t - 140, t + 10)
            return artists

        num_frames = 500
        step_skip = max(1, total_steps // num_frames)
        ani = animation.FuncAnimation(
            fig, animate, frames=range(0, total_steps, step_skip),
            interval=100, blit=False,
        )

        plt.tight_layout()

        if save_path:
            if save_path.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=15, bitrate=1800)
                ani.save(save_path, writer=writer)
            else:
                ani.save(save_path, writer='pillow', fps=15)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()


# ============================================================
# BS site / sector orientation defaults
# ------------------------------------------------------------
# These match the MATLAB reference: 7 sites in a hexagonal
# cluster centred at (300, 300) with InterBSHSpacing = 200 m.
# Sectors share the 3GPP convention (30 deg, 150 deg, 270 deg).
# Replace via constructor args when a .mat with the exact
# orientations is available.
# ============================================================
def get_bs_sites() -> np.ndarray:
    return np.array([
        [300.0, 300.0],   # BS 0 — centre
        [300.0, 500.0],   # BS 1 — top
        [126.8, 400.0],   # BS 2 — upper left
        [126.8, 200.0],   # BS 3 — lower left
        [300.0, 100.0],   # BS 4 — bottom
        [473.2, 200.0],   # BS 5 — lower right
        [473.2, 400.0],   # BS 6 — upper right
    ])


def get_bs_positions() -> np.ndarray:
    """Backwards-compatible 21-sector positions (one entry per global sector).

    Each sector entry equals the BS site coordinate. The dashboard no
    longer offsets sectors visually — orientation is conveyed via the
    boresight line and antenna pattern instead.
    """
    sites = get_bs_sites()
    return np.repeat(sites, 3, axis=0)


if __name__ == "__main__":
    bs_sites = get_bs_sites()
    dash = MobilityDashboard(bs_sites)
    steps = 1000
    history = {
        'ue_pos': [[300 + t * 0.2, 300 + t * 0.1] for t in range(steps)],
        'serving_bs': [0 if t < 500 else 3 * 3 for t in range(steps)],
        'rsrp': np.random.uniform(-100, -60, (steps, 21)).tolist(),
        'mcs': np.random.uniform(0, 5.5, (steps, 21)).tolist(),
    }
    os.makedirs("results/animations", exist_ok=True)
    dash.render_episode(history, save_path="results/animations/dashboard_video_test.mp4")
