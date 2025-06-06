import logging
import hydra
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf

from models.utils import set_seed
from models import (
    GPT,
    GPTConfig,
    RADT,
    RADTConfig,
    Starformer,
    StarformerConfig,
    DC,
    DCConfig,
    Trainer,
    TrainerConfig,
)
from create_dataset import create_dataset
from eval_dt_atari import evaluate


class StateActionReturnDataset(Dataset):
    def __init__(
        self,
        data,
        block_size,
        actions,
        returns,
        done_idxs,
        rtgs,
        timesteps,
        model_type="dt",
    ):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.returns = returns
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.model_type = model_type

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(
            np.array(self.data[idx:done_idx]), dtype=torch.float32
        ).reshape(block_size, -1)  # (block_size, 4*84*84)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(
            1
        )  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        if self.model_type in ["dt", "radt", "dc"] or "star" in self.model_type:
            timesteps = torch.tensor(
                self.timesteps[idx : idx + 1], dtype=torch.int64
            ).unsqueeze(1)

        return states, actions, rtgs, timesteps


@hydra.main(version_base=None, config_path="conf", config_name="default")
def experiment(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    cfg.log_dir = cfg.paths.output_dir

    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(
        cfg.num_buffers,
        cfg.num_steps,
        cfg.game,
        cfg.data_dir_prefix,
        cfg.trajectories_per_buffer,
    )

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    train_dataset = StateActionReturnDataset(
        obss,
        cfg.seq_len * 3,
        actions,
        returns,
        done_idxs,
        rtgs,
        timesteps,
        model_type=cfg.model_type,
    )
    if cfg.model_type == "radt":
        mconf = RADTConfig(
            train_dataset.vocab_size,
            cfg.seq_len * 3,
            n_layer=cfg.n_layers,
            n_head=8,
            n_embd=128,
            model_type=cfg.model_type,
            max_timestep=max(timesteps),
            seq_len=cfg.seq_len,
            stepra=cfg.stepra,
            alpha_scale=cfg.alpha_scale,
            seqra=cfg.seqra,
            rtg_scale=cfg.rtg_scale,
            radt_proj=cfg.radt_proj,
            pe_sinusoid=cfg.pe_sinusoid,
        )
        model = RADT(mconf)
    elif "star" in cfg.model_type:
        mconf = StarformerConfig(
            train_dataset.vocab_size,
            img_size=(4, 84, 84),
            patch_size=(7, 7),
            pos_drop=0.1,
            resid_drop=0.1,
            N_head=8,
            D=192,
            local_N_head=4,
            local_D=64,
            model_type=cfg.model_type,
            n_layer=6,
            C=4,
            maxT=cfg.seq_len,
            seq_len=cfg.seq_len,
        )
        model = Starformer(mconf)
    elif "dc" in cfg.model_type:
        mconf = DCConfig(
            train_dataset.vocab_size,
            cfg.seq_len * 3,
            conv_proj=cfg.dc_proj,
            seq_len=cfg.seq_len,
            n_layer=6,
            n_head=8,
            n_embd=128,
            model_type=cfg.model_type,
            max_timestep=max(timesteps),
            token_mixer="conv",
            window_size=6,
        )
        model = DC(mconf)
    elif cfg.model_type == "dt":
        mconf = GPTConfig(
            train_dataset.vocab_size,
            train_dataset.block_size,
            seq_len=cfg.seq_len,
            n_layer=cfg.n_layers,
            n_head=8,
            n_embd=128,
            model_type=cfg.model_type,
            max_timestep=max(timesteps),
        )
        model = GPT(mconf)
    else:
        raise NotImplementedError

    # initialize a trainer instance and kick off training
    epochs = cfg.epochs
    tconf = TrainerConfig(
        max_epochs=epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * cfg.seq_len * 3,
        num_workers=4,
        seed=cfg.seed,
        model_type=cfg.model_type,
        game=cfg.game,
        max_timestep=max(timesteps),
        vocab_size=train_dataset.vocab_size,
        args=cfg,
    )
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    evaluate(
        cfg.seed,
        cfg.log_dir,
        "best",
        cfg.target_returns["test"],
        "",
        cfg.num_test_episodes,
    )


if __name__ == "__main__":
    experiment()
