import csv
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from tqdm.auto import trange


class Trainer:
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        target_returns,
        scheduler=None,
        eval_fns=None,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.target_returns = target_returns
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.eval_fns = [] if eval_fns is None else eval_fns

        if self.cfg.validation_mode == "align":
            self.selected_validation_score = np.inf
        elif self.cfg.validation_mode == "best":
            self.selected_validation_score = -np.inf
        self.selected_idx = 0

    def train_iteration(self, num_steps, epoch_idx=1):
        if epoch_idx != 0:  # skip first epoch
            # train
            self.model.train()
            train_losses = []
            for i in trange(0, num_steps, desc="train"):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        # validation
        self.model.eval()
        validation_score = 0
        eval_scores = []
        for target_return, eval_fn in zip(self.target_returns, self.eval_fns):
            returns, lengths = eval_fn(self.model)
            mean_return = np.mean(returns)
            eval_scores.append(mean_return)
            if self.cfg.validation_mode == "align":
                validation_score += np.abs(mean_return - target_return)
        with open(
            Path(self.cfg.paths.output_dir) / "train_result.csv", "a", newline=""
        ) as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [epoch_idx * num_steps]
                + eval_scores
                + [
                    datetime.now(timezone.utc)
                    .astimezone(ZoneInfo("Asia/Tokyo"))
                    .strftime("%Y-%m-%d %H:%M:%S")
                ]
            )

        if self.cfg.validation_mode == "best":
            validation_score = max(eval_scores)

        # save checkpoint
        checkpoint_dir = Path(self.cfg.paths.output_dir) / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step": epoch_idx * num_steps,
        }
        if (
            self.cfg.validation_mode == "align"
            and self.selected_validation_score > validation_score
        ) or (
            self.cfg.validation_mode == "best"
            and self.selected_validation_score <= validation_score
        ):
            self.selected_idx = epoch_idx * num_steps
            self.selected_validation_score = validation_score
            torch.save(ckpt, checkpoint_dir / "best.pth")
        torch.save(ckpt, checkpoint_dir / f"{epoch_idx * num_steps}.pth")
