import yaml
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf

# make deterministic
from models.utils import set_seed

import numpy as np
import torch

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


def evaluate(seed, path, suffix, returns, name, eval_episodes):
    set_seed(seed)

    hydra_config_path = Path(path) / ".hydra/config.yaml"
    if hydra_config_path.exists():
        config = OmegaConf.load(hydra_config_path)
    else:
        config_path = Path(path) / "config.yaml"
        with open(config_path) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
    checkpint_path = Path(path).joinpath("checkpoint").joinpath(f"model_{suffix}.pth")
    if not checkpint_path.exists():
        raise ValueError(f"checkpoint {checkpint_path} not exists!")
    checkpint = torch.load(checkpint_path, weights_only=False)
    max_timestep = checkpint["max_timestep"]

    file_name = "results"
    if name:
        file_name += f"_{name}"
    f_r = open(Path(path).joinpath(f"{file_name}.txt"), "w")
    res_json_path = Path(path).parent.joinpath(f"{file_name}.json")
    if res_json_path.exists():
        with open(res_json_path, "r") as f:
            res_dict = json.load(f)
        res_dict[config["seed"]] = {}
    else:
        res_dict = {config["seed"]: {}}

    if config["game"] == "Breakout":
        VOCAB_SIZE = 4
    elif config["game"] == "Pong":
        VOCAB_SIZE = 6
    elif config["game"] == "Qbert":
        VOCAB_SIZE = 6
    elif config["game"] == "Seaquest":
        VOCAB_SIZE = 18

    if config["model_type"] == "dt":
        mconf = GPTConfig(
            VOCAB_SIZE,
            config["seq_len"] * 3,
            seq_len=config["seq_len"],
            n_layer=config["n_layers"],
            n_head=8,
            n_embd=128,
            model_type=config["model_type"],
            max_timestep=max_timestep,
        )
        model = GPT(mconf)
    elif config["model_type"] == "radt":
        mconf = RADTConfig(
            VOCAB_SIZE,
            config["seq_len"] * 3,
            n_layer=config["n_layers"],
            n_head=8,
            n_embd=128,
            model_type=config["model_type"],
            max_timestep=max_timestep,
            seq_len=config["seq_len"],
            stepra=config["stepra"],
            alpha_scale=config["alpha_scale"],
            seqra=config["seqra"],
            rtg_scale=config["rtg_scale"],
            radt_proj=config["radt_proj"],
            pe_sinusoid=config["pe_sinusoid"],
        )
        model = RADT(mconf)
    elif config["model_type"] == "dc":
        mconf = DCConfig(
            VOCAB_SIZE,
            config["seq_len"] * 3,
            conv_proj=config["dc_proj"],
            n_layer=6,
            n_head=8,
            n_embd=128,
            model_type=config["model_type"],
            max_timestep=max_timestep,
            token_mixer="conv",
            window_size=6,
            seq_len=config["seq_len"],
        )
        model = DC(mconf)
    elif "star" in config["model_type"]:
        mconf = StarformerConfig(
            VOCAB_SIZE,
            img_size=(4, 84, 84),
            patch_size=(7, 7),
            pos_drop=0.1,
            resid_drop=0.1,
            N_head=8,
            D=192,
            local_N_head=4,
            local_D=64,
            model_type=config["model_type"],
            n_layer=6,
            C=4,
            maxT=config["seq_len"],
            seq_len=config["seq_len"],
        )
        model = Starformer(mconf)

    model.load_state_dict(checkpint["model_state_dict"])

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        max_epochs=0,
        batch_size=1,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * 1 * config["seq_len"] * 3,
        num_workers=4,
        seed=seed,
        model_type=config["model_type"],
        game=config["game"],
        max_timestep=max_timestep,
        vocab_size=VOCAB_SIZE,
        args=None,
    )
    trainer = Trainer(model, None, None, tconf)

    if returns is None:
        if "target_returns" in config:
            rets = config["target_returns"]["test"]
        else:
            game_config_path = f"./conf/experiment/{config['game']}.yaml"
            with open(game_config_path) as f:
                game_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            rets = game_config["target_returns"]["test"]
    else:
        rets = returns

    for target_return in rets:
        real_rets, eval_timestep_len = trainer.get_returns(
            int(target_return),
            eval_episodes=eval_episodes,
            video_save_dir_path=Path(path) / "videos",
        )
        eval_return, eval_std = np.mean(real_rets), np.std(real_rets)
        print_message = f"Target return: {target_return}, Eval return: {eval_return:.2f} ± {eval_std:.2f}, timestep: {eval_timestep_len}"
        print(print_message, file=f_r, flush=True)
        res_dict[config["seed"]].update({target_return: real_rets})

    with open(res_json_path, "w") as f:
        json.dump(res_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--path", type=str)
    parser.add_argument("--suffix", type=str, default="best")
    parser.add_argument("--returns", nargs="+", type=float)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--eval_episodes", type=int, default=10)
    args = parser.parse_args()

    evaluate(**vars(args))
