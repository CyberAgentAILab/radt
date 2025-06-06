import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import trange

import csv
import pickle
import random
from pathlib import Path

from evaluate import evaluate
from evaluation.evaluate_episodes import evaluate_episode_rtg
from common.utils import (
    discount_cumsum,
    get_env_info,
    get_model_optimizer,
    get_trainer,
    seed_all,
)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def experiment(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed_all(cfg.seed)
    device = cfg.device

    env_name, dataset = cfg.env, cfg.dataset

    env, max_ep_len, scale = get_env_info(env_name, dataset)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    if dataset == "medium-expert":
        dataset_path = f"{cfg.paths.data_dir}/{env_name}-expert-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        dataset_path = f"{cfg.paths.data_dir}/{env_name}-medium-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    else:
        dataset_path = f"{cfg.paths.data_dir}/{env_name}-{dataset}-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

    if "star" in cfg.model_type:
        trajectories = [traj for traj in trajectories if traj["rewards"].shape[0] > 1]

    # save all path information into separate lists
    mode = cfg.mode
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")

    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    K = cfg.K
    batch_size = cfg.batch_size
    pct_traj = cfg.pct_traj

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    if cfg.timestep_sampling:
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    else:
        p_sample = None

    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            if "star" in cfg.model_type:
                si = random.randint(1, traj["rewards"].shape[0] - 1)
            else:
                si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            if "star" in cfg.model_type:
                a.append(traj["actions"][si - 1 : si + max_len].reshape(1, -1, act_dim))
            else:
                a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            if "star" in cfg.model_type:
                tlen = a[-1].shape[1] - 1
                s[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen, state_dim)), s[-1][:, :tlen]], axis=1
                )
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate(
                    [
                        np.ones((1, max_len + 1 - a[-1].shape[1], act_dim)) * -10.0,
                        a[-1],
                    ],
                    axis=1,
                )
                r[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen, 1)), r[-1][:, :tlen]], axis=1
                )
                d[-1] = np.concatenate(
                    [np.ones((1, max_len - tlen)) * 2, d[-1][:, :tlen]], axis=1
                )
                rtg[-1] = (
                    np.concatenate(
                        [np.zeros((1, max_len - tlen, 1)), rtg[-1][:, : tlen + 1]],
                        axis=1,
                    )
                    / scale
                )
                timesteps[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen)), timesteps[-1][:, :tlen]], axis=1
                )
                mask.append(
                    np.concatenate(
                        [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                    )
                )
            else:
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
                )
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate(
                    [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
                )
                r[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1
                )
                d[-1] = np.concatenate(
                    [np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1
                )
                rtg[-1] = (
                    np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                    / scale
                )
                timesteps[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
                )
                mask.append(
                    np.concatenate(
                        [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                    )
                )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target):
        def fn(model):
            returns, lengths = [], []
            for _ in trange(
                0, cfg.num_eval_episodes, desc=f"eval_{target}", leave=False
            ):
                with torch.no_grad():
                    ret, length, _, _ = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target / scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        clip_action=cfg.clip_action,
                    )
                returns.append(ret)
                lengths.append(length)
            return returns, lengths

        return fn

    model, optimizer, scheduler = get_model_optimizer(
        cfg, state_dim, act_dim, max_ep_len, device
    )
    loss_fn = lambda a_hat, a: torch.mean((a_hat - a) ** 2)

    if len(cfg.eval_targets) == 0:
        cfg.eval_targets = cfg.target_returns.eval

    trainer = get_trainer(
        cfg=cfg,
        model_type=cfg.model_type,
        model=model,
        batch_size=batch_size,
        get_batch=get_batch,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=[eval_episodes(tar) for tar in cfg.eval_targets],
        target_returns=cfg.eval_targets,
    )

    # train + validation
    epoch_num = cfg.num_iterations // cfg.eval_every
    print("Training Start!")
    with open(Path(cfg.paths.output_dir) / "train_result.csv", "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["step"]
            + [float(target_return) for target_return in cfg.eval_targets]
            + ["date"]
        )
    for epoch_idx in range(
        epoch_num + 1
    ):  # +1 for evaluation before training (epoch_idx=0)
        if epoch_idx != 0:
            print(f"Epoch: {epoch_idx}/{epoch_num}")
        trainer.train_iteration(num_steps=cfg.eval_every, epoch_idx=epoch_idx)

    # test
    target_returns = [float(t) for t in cfg.target_returns.test]
    evaluate(
        cfg,
        model,
        cfg.num_test_episodes,
        target_returns,
        env,
        state_dim,
        act_dim,
        max_ep_len,
        scale,
        state_mean,
        state_std,
        device,
        input_dir=Path(cfg.paths.output_dir),
        suffix="best",
        name="",
    )


if __name__ == "__main__":
    experiment()
