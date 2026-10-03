import argparse
import os
import sys

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config
from src.distrl.viz.dashboard import MobilityDashboard, get_bs_sites


def _build_agent(agent_type: str, env: LTMEnv, device: str):
    agent_type = agent_type.lower()
    config = Config.get()
    agent_cfg = config['agent']
    if agent_type == "dqn":
        return DQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    if agent_type == "qrdqn":
        return QRDQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    if agent_type == "ltm_baseline":
        return LTMBaselineAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    raise ValueError(f"Unknown agent type: {agent_type}")


def run_agent_dashboard(
    agent_type: str = "dqn",
    ue_idx: int = 0,
    model_path: str | None = None,
    output_path: str | None = None,
) -> None:
    print(
        f"=== Generating Dashboard for {agent_type.upper()} "
        f"(file index {ue_idx} → User{ue_idx + 1}_precomputed.npz) ==="
    )

    env = LTMEnv(config=Config.get())
    env.current_ue_idx = ue_idx

    is_baseline = agent_type.lower() == "ltm_baseline"
    agent = _build_agent(agent_type, env, device="cpu")

    if not is_baseline:
        if not model_path:
            model_path = f"results/models/{agent_type}_best.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}.")
            return
        agent.load(model_path)
        print(f"Loaded model from {model_path}")

    state, _ = env.reset()
    done = False
    history: dict[str, list] = {
        'ue_pos': [], 'serving_bs': [], 'rsrp': [], 'mcs': []
    }

    print("Running episode...")
    while not done:
        t_idx = min(env.t, env.Max_iter - 1)
        history['ue_pos'].append(env.ue_positions[t_idx].tolist())
        history['serving_bs'].append(int(env.ServingBSSector[t_idx]))
        history['rsrp'].append(env.PL3[:, t_idx].tolist())
        history['mcs'].append(env.all_mcs_episode[:, t_idx].tolist())

        if is_baseline:
            state, _, done, _, _ = env.step(
                0, high_res_callback=agent.select_action
            )
        else:
            action = agent.select_action(state, epsilon=0)
            state, _, done, _, _ = env.step(action)

    print(f"Episode finished at t={env.t}. Generating animation...")

    bs_sites = get_bs_sites()
    dash = MobilityDashboard(bs_sites)

    if not output_path:
        if model_path and "results/benchmarks" in model_path:
            bmk_dir = os.path.dirname(os.path.dirname(model_path))
            output_path = os.path.join(
                bmk_dir, "animations", f"{agent_type}_ue{ue_idx}_final.mp4"
            )
        else:
            output_path = f"results/animations/{agent_type}_ue{ue_idx}_final.mp4"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dash.render_episode(history, save_path=output_path)
    print(f"Animation saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, default="dqn",
        choices=["dqn", "qrdqn", "ltm_baseline"],
    )
    parser.add_argument(
        "--ue_idx", type=int, default=500,
        help=(
            "1-based UE number (1..1000); the script converts to 0-based "
            "internally, so --ue_idx 42 loads User42_precomputed.npz and "
            "the output filename embeds 'ue41'."
        ),
    )
    parser.add_argument(
        "--model_path", type=str,
        help="Specific path to model weights (ignored for ltm_baseline)",
    )
    parser.add_argument("--output_path", type=str, help="Where to save the animation")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config (loaded into the Config singleton)",
    )
    args = parser.parse_args()

    Config.set_config_path(args.config)
    run_agent_dashboard(
        args.agent, args.ue_idx - 1,
        args.model_path, args.output_path,
    )
