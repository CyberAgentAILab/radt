"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import time
import csv
import json
from pathlib import Path
from collections import deque

import atari_py
import random
import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from models.utils import sample

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    args = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.data_len = 0

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            self.data_len = len(loader)

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y, r, t) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    if "star" in self.config.model_type:
                        a = torch.cat(
                            [
                                torch.ones(
                                    y.size(0), 1, 1, device=y.device, dtype=torch.long
                                )
                                * config.vocab_size,
                                y[:, :-1],
                            ],
                            dim=1,
                        )
                        logits, loss = model(x, a, y, r, t)
                    else:
                        logits, loss = model(x, y, y, r, t)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        self.tokens = 0  # counter used for learning rate decay

        if self.config.model_type == "naive":
            target_returns = [0]
        elif (
            self.config.model_type in ["dt", "radt", "dc"]
            or "star" in self.config.model_type
        ):
            target_returns = self.config.args.target_returns.eval
        else:
            raise NotImplementedError()

        best_err = np.inf

        record_csv = str(Path(self.config.args.log_dir) / "records.csv")
        with open(record_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Epoch", "Training Time (s)", "Validation Time (s)"] + target_returns
            )

        epoch_raw_result_dir = Path(self.config.args.log_dir) / "raw_results"
        epoch_raw_result_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(config.max_epochs):
            start_train = time.time()
            run_epoch("train", epoch_num=epoch)
            train_time = time.time() - start_train

            ret_err = []
            record_results = {}
            raw_results = {}
            start_val = time.time()
            for t_r in target_returns:
                real_rets, eval_timestep_len = self.get_returns(
                    t_r, eval_episodes=self.config.args.num_eval_episodes
                )
                eval_return = np.mean(real_rets)
                ret_err.append(np.abs(eval_return - t_r))
                record_results[t_r] = eval_return
                raw_results[t_r] = real_rets
            val_time = time.time() - start_val
            with open(epoch_raw_result_dir / f"epoch_{epoch}.json", "w") as f:
                json.dump(raw_results, f)

            save_dir = Path(self.config.args.log_dir).joinpath("checkpoint")
            save_dir.mkdir(parents=True, exist_ok=True)
            fn = save_dir.joinpath("model_{}.pth".format(epoch + 1))
            checkpoint = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "max_timestep": self.config.max_timestep,
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, fn)

            sum_err = np.sum(ret_err)
            if sum_err < best_err:
                best_err = sum_err
                fn = save_dir.joinpath("model_best.pth")
                checkpoint = {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "max_timestep": self.config.max_timestep,
                    "epoch": epoch + 1,
                }
                torch.save(checkpoint, fn)

            with open(record_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [epoch, train_time, val_time]
                    + [record_results[t_r] for t_r in target_returns]
                )

    def get_returns(
        self, ret, eval_episodes: int = 10, video_save_dir_path: Path = None
    ):
        ret = 100
        self.model.train(False)
        args = Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_timesteps = [], []
        done = True
        for i in tqdm(range(eval_episodes)):
            frames = []

            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            frames.append(self.get_rgb_frame(env))
            current_rtg = ret
            rtgs = [current_rtg]
            actions = []
            # first state is from env, first rtg is target return, and first timestep is 0
            if "star" in self.config.model_type:
                actions += (
                    torch.ones(1, 1, device=self.device, dtype=torch.long)
                    * self.config.vocab_size
                )
            sampled_action = sample(
                self.model.module,
                state,
                1,
                temperature=1.0,
                sample=True,
                actions=torch.tensor(actions, dtype=torch.long)
                .to(self.device)
                .unsqueeze(1)
                .unsqueeze(0),
                rtgs=torch.tensor(rtgs, dtype=torch.long)
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(-1),
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
                model_type=self.config.model_type,
            )

            j = 0
            all_states = state
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [sampled_action]
                state, reward, done, life_termination = env.step(action)
                frames.append(self.get_rgb_frame(env))
                reward_sum += reward
                j += 1

                if life_termination:
                    print(
                        f"episode {i}, step {j}, action {action}, return {reward_sum}, done {done}, life_termination {life_termination}"
                    )

                if done or j >= self.config.max_timestep:
                    T_rewards.append(reward_sum)
                    T_timesteps.append(j)
                    done = True

                    if video_save_dir_path is not None:
                        video_save_dir_path.mkdir(parents=True, exist_ok=True)
                        video_save_path = video_save_dir_path / f"episode_{i}.mp4"
                        with imageio.get_writer(video_save_path, fps=30) as writer:
                            for f in frames:
                                writer.append_data(np.asarray(f))
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                if life_termination:
                    current_rtg = ret
                    rtgs = [current_rtg]
                    actions = []
                    all_states = state
                    j = 0
                else:
                    current_rtg = current_rtg - reward
                    rtgs += [current_rtg]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                if (
                    self.config.model_type in ["dt", "radt", "dc"]
                    or "star" in self.config.model_type
                ):
                    timesteps = min(j, self.config.max_timestep) * torch.ones(
                        (1, 1, 1), dtype=torch.int64
                    ).to(self.device)
                sampled_action = sample(
                    self.model.module,
                    all_states.unsqueeze(0),
                    1,
                    temperature=1.0,
                    sample=True,
                    actions=torch.tensor(actions, dtype=torch.long)
                    .to(self.device)
                    .unsqueeze(1)
                    .unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.long)
                    .to(self.device)
                    .unsqueeze(0)
                    .unsqueeze(-1),
                    timesteps=timesteps,
                    model_type=self.config.model_type,
                )

        eval_timestep_len = np.mean(T_timesteps)
        print(
            f"target return: {ret}, eval return: {np.mean(T_rewards)} ± {np.std(T_rewards)}, eval timestep: {eval_timestep_len}"
        )
        self.model.train(True)
        return T_rewards, eval_timestep_len

    def get_rgb_frame(self, e):
        return e.render("rgb_array")


class Env:
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt("random_seed", args.seed)
        self.ale.setInt("max_num_frames_per_episode", args.max_episode_length)
        self.ale.setFloat("repeat_action_probability", 0)  # Disable sticky actions
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.loadROM(
            atari_py.get_game_path(args.game)
        )  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = (
            False  # Used to check if resetting only from loss of life
        )
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(
            self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR
        )
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break

        lives = self.ale.lives()
        if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
            life_termination = True
        else:
            life_termination = False
        self.lives = lives

        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done, life_termination

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self, mode: str = "rgb_array"):
        frame = self.ale.getScreenRGB()  # shape (210, 160, 3)  BGR

        if mode == "rgb_array":
            return frame

        if mode == "human":
            cv2.imshow("screen", frame[:, :, ::-1])
            cv2.waitKey(1)
            return frame

        raise ValueError(f"Unsupported render mode: {mode!r}")

    def close(self):
        cv2.destroyAllWindows()


class Args:
    def __init__(self, game, seed):
        self.device = torch.device("cuda")
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
