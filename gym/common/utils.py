import torch
import numpy as np
import gym
import os
import random

from models import (
    RADT,
    Starformer,
    StarformerConfig,
    DecisionConvFormer,
    DecisionTransformer,
)

from training import (
    RADTTrainer,
    StarformerTrainer,
    DecisionConvFormerTrainer,
    DecisionTransformerTrainer,
)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device="cuda")
    return x


def get_env_info(env_name, dataset):
    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        max_ep_len = 1000
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v3")
        max_ep_len = 1000
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("Walker2d-v3")
        max_ep_len = 1000
        scale = 1000.0
    elif env_name == "ant":
        env = gym.make("Ant-v3")
        max_ep_len = 1000
        scale = 1000.0
    elif env_name == "antmaze":
        env = gym.make(f"{env_name}-{dataset}-v2")
        max_ep_len = 1000
        scale = 1.0
    else:
        raise NotImplementedError

    return env, max_ep_len, scale


def get_model_optimizer(cfg, state_dim, act_dim, max_ep_len, device):
    if cfg.model_type == "radt":
        model = RADT(
            state_dim=state_dim,
            act_dim=act_dim,
            seq_len=cfg.K,
            episode_len=max_ep_len,
            embedding_dim=cfg.embed_dim,
            num_layers=cfg.n_layer,
            num_heads=cfg.n_head,
            attention_dropout=cfg.dropout,
            residual_dropout=cfg.dropout,
            use_learnable_pos_emb=cfg.radt.use_learnable_pos_emb,
            stepra=cfg.radt.stepra,
            alpha_scale=cfg.radt.alpha_scale,
            seqra=cfg.radt.seqra,
            remove_act_embs=cfg.remove_act_embs,
            action_tanh=cfg.radt.action_tanh,
        )
    elif cfg.model_type == "star_rwd_rtg":
        star_conf = StarformerConfig(
            act_dim,
            context_length=cfg.K,
            pos_drop=cfg.dropout,
            resid_drop=cfg.dropout,
            N_head=cfg.n_head,
            D=cfg.embed_dim,
            local_N_head=4,
            local_D=16,
            model_type=cfg.model_type,
            max_timestep=1000,
            n_layer=cfg.n_layer,
            maxT=cfg.K,
            T_each_level=None,
            state_dim=state_dim,
            max_action=act_dim,
            episode_len=max_ep_len,
            action_dim=act_dim,
            seq_len=cfg.K,
        )
        model = Starformer(star_conf)
    elif cfg.model_type == "dc":
        model = DecisionConvFormer(
            env_name=cfg.env,
            dataset=cfg.dataset,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=cfg.K,
            max_ep_len=max_ep_len,
            remove_act_embs=cfg.remove_act_embs,
            hidden_size=cfg.embed_dim,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_inner=4 * cfg.embed_dim,
            drop_p=cfg.dropout,
            window_size=cfg.dc.conv_window_size,
            activation_function=cfg.activation_function,
            resid_pdrop=cfg.dropout,
        )
    elif cfg.model_type == "dt":
        model = DecisionTransformer(
            env_name=cfg.env,
            dataset=cfg.dataset,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=cfg.K,
            max_ep_len=max_ep_len,
            remove_act_embs=cfg.remove_act_embs,
            hidden_size=cfg.embed_dim,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_inner=4 * cfg.embed_dim,
            activation_function=cfg.activation_function,
            resid_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
        )
    else:
        raise NotImplementedError
    model = model.to(device=device)

    warmup_steps = cfg.warmup_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    return model, optimizer, scheduler


def get_trainer(model_type, **kwargs):
    if model_type == "radt":
        return RADTTrainer(**kwargs)
    elif model_type == "star_rwd_rtg":
        return StarformerTrainer(**kwargs)
    elif model_type == "dc":
        return DecisionConvFormerTrainer(**kwargs)
    elif model_type == "dt":
        return DecisionTransformerTrainer(**kwargs)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def seed_all(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
