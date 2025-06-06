import json
import random
import pickle
from pathlib import Path

import numpy as np
import torch
from d4rl import infos
from tqdm.auto import trange

from common.utils import get_env_info, get_model_optimizer
from evaluation.evaluate_episodes import evaluate_episode_rtg


def evaluate(
    cfg,
    model,
    num_episodes,
    target_returns,
    env,
    state_dim,
    act_dim,
    max_ep_len,
    scale,
    state_mean,
    state_std,
    device,
    input_dir: Path,
    suffix="best",
    name="",
    render_path="",
):
    print("Test Start!")

    results_dir = input_dir
    if name != "":
        results_dir = results_dir / name
        results_dir.mkdir(parents=True, exist_ok=True)
        name = f"_{name}"

    # create result files
    if "single-run" in input_dir.as_posix():
        result_json_path = input_dir / f"result{name}.json"
    else:
        result_json_path = input_dir.parent / f"result{name}.json"
    if result_json_path.exists():
        with open(result_json_path, "r") as f:
            results_dict = json.load(f)
        results_dict[cfg.seed] = {}
    else:
        results_dict = {cfg.seed: {}}
    result_text_file = open(results_dir / "result.txt", "w")

    # load model
    best_checkpoint_path = input_dir / "checkpoint" / f"{suffix}.pth"
    print(f"Load from {best_checkpoint_path}", file=result_text_file, flush=True)
    best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"], strict=False)
    print(f"Best Step: {best_checkpoint['step']}", file=result_text_file, flush=True)
    model.eval()

    return_min = infos.REF_MIN_SCORE[f"{cfg.env}-{cfg.dataset}-v2"]
    return_max = infos.REF_MAX_SCORE[f"{cfg.env}-{cfg.dataset}-v2"]

    for target in target_returns:
        returns, lengths = [], []
        target_returns, rewards_list = [], []
        for episode_id in trange(0, num_episodes, desc=f"test_{target}"):
            episode_render_path = None
            if render_path:
                episode_render_path = render_path / f"{episode_id}.mp4"
            with torch.no_grad():
                ret, length, target_return, rewards = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target / scale,
                    mode=cfg.mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    clip_action=cfg.clip_action,
                    render_path=episode_render_path,
                )
            returns.append(ret)
            lengths.append(length)
            target_returns.append(target_return)
            rewards_list.append(rewards)
        mean_return = np.mean(returns)
        normalized_mean_return = (
            (mean_return - return_min) * 100 / (return_max - return_min)
        )
        print_message = f"Target return {target}: actual return {mean_return:.2f} (Normalized: {normalized_mean_return:.2f})"
        print(print_message, file=result_text_file, flush=True)
        results_dict[cfg.seed].update({target: returns})

    result_text_file.close()
    with open(result_json_path, "w") as f:
        json.dump(results_dict, f)

    print("Test Finished!")


def load_evaluation_infos(args):
    cfg = OmegaConf.load(Path(args.path) / ".hydra/config.yaml")

    if args.target_returns is not None and len(args.target_returns) > 0:
        target_returns = args.target_returns
    else:
        target_returns = cfg.target_returns.test
        target_returns = [float(t) for t in target_returns]

    if args.num_test_episodes != 0:
        cfg.num_test_episodes = args.num_test_episodes

    device = cfg.device

    env, max_ep_len, scale = get_env_info(cfg.env, cfg.dataset)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # state_mean, state_std
    if cfg.dataset == "medium-expert":
        dataset_path = (
            f"{cfg.paths.data_dir}/{cfg.env}-expert-v2.pkl"  # TODO: unuse cfg.paths
        )
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        dataset_path = f"{cfg.paths.data_dir}/{cfg.env}-medium-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    else:
        dataset_path = f"{cfg.paths.data_dir}/{cfg.env}-{cfg.dataset}-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
    states = []
    for path in trajectories:
        states.append(path["observations"])
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    model, _, _ = get_model_optimizer(cfg, state_dim, act_dim, max_ep_len, device)

    base_dir = Path(args.path)
    if args.eval_name != "":
        base_dir = Path(args.path) / args.eval_name

    render_path = ""
    if args.vis_render:
        render_path = base_dir / "render"
        render_path.mkdir(parents=True, exist_ok=True)

    return {
        "cfg": cfg,
        "num_episodes": cfg.num_test_episodes,
        "target_returns": target_returns,
        "env": env,
        "max_ep_len": max_ep_len,
        "scale": scale,
        "state_dim": state_dim,
        "act_dim": act_dim,
        "state_mean": state_mean,
        "state_std": state_std,
        "model": model,
        "device": device,
        "input_dir": Path(args.path),
        "suffix": args.suffix,
        "name": args.eval_name,
        "render_path": render_path,
    }


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="best")  # step or "best"
    parser.add_argument("--target_returns", nargs="+", type=float)
    parser.add_argument("--eval_name", type=str, default="")
    parser.add_argument("--num_test_episodes", type=int, default=0)
    parser.add_argument("--vis_render", action="store_true")
    args = parser.parse_args()

    eval_infos = load_evaluation_infos(args)
    evaluate(**eval_infos)
