import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from einops import rearrange

logger = logging.getLogger(__name__)


class RADTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_position_angle_vec(position, d_hid):
    return [
        position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)
    ]


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        seq_len=20,
        q_len=2,
        kv_len=2,
        cross=False,
        radt_proj=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim**-0.5
        self.q_len = q_len
        self.kv_len = kv_len
        self.cross = cross

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.radt_proj = radt_proj
        if self.radt_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        if not self.cross:
            seq_len = seq_len * q_len
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8)).view(
                1, 1, seq_len, seq_len
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def forward(self, q, k, v, padding_mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.cross:
            nd, ns = dots.size(-2) // self.q_len, dots.size(-1) // self.kv_len
            mask = self.bias[:, :, ns - nd : ns, :ns]
            mask = mask.repeat_interleave(self.q_len, dim=2).repeat_interleave(
                self.kv_len, dim=3
            )
        else:
            nd, ns = dots.size(-2), dots.size(-1)
            mask = self.bias[:, :, ns - nd : ns, :ns]
        dots = torch.where(mask.bool(), dots, self.masked_bias.to(dots.dtype))

        if padding_mask is not None:
            k_padding_mask = padding_mask.repeat_interleave(self.kv_len, dim=1)
            k_padding_mask = k_padding_mask[:, None, None, :]
            k_padding_mask = k_padding_mask.to(dtype=dots.dtype)
            k_padding_mask = k_padding_mask * -10000.0
            dots = dots + k_padding_mask

        attn = self.softmax(dots)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.radt_proj:
            out = self.proj(out)
            out = self.proj_drop(out)
        return out, attn


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        seq_len=10,
        q_len=0,
        kv_len=0,
        stepra=False,
        alpha_scale=False,
        seqra=False,
        radt_proj=True,
    ):
        super().__init__()
        self.stepra = stepra
        self.alpha_scale = alpha_scale
        self.seqra = seqra

        if self.stepra:
            module_num = 6 if self.seqra else 4
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * module_num))
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            if self.seqra:
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = nn.LayerNorm(dim)
            if self.seqra:
                self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)

        if self.alpha_scale and self.seqra:
            self.fc_alpha_ca = nn.Linear(dim * 2, dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            seq_len=seq_len,
            q_len=q_len,
            kv_len=q_len,
            cross=False,
            radt_proj=radt_proj,
        )

        if self.seqra:
            self.norm_k = nn.LayerNorm(dim)
            self.norm_v = nn.LayerNorm(dim)
            self.seqra = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                seq_len=seq_len,
                q_len=q_len,
                kv_len=kv_len,
                cross=True,
                radt_proj=radt_proj,
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop
        )

    def forward(self, q, c=None, pe=None, padding_mask=None):
        if self.stepra:
            if self.seqra:
                scale_sa, shift_sa, scale_ca, shift_ca, scale_mlp, shift_mlp = (
                    self.modulation(c).chunk(6, dim=2)
                )
            else:
                scale_sa, shift_sa, scale_mlp, shift_mlp = self.modulation(c).chunk(
                    4, dim=2
                )

        q1, attn = self.attn(q, q, q, padding_mask)
        q = q1 + q
        if self.stepra:
            scale_sa = scale_sa.repeat_interleave(
                q.shape[1] // scale_sa.shape[1], dim=1
            )
            shift_sa = shift_sa.repeat_interleave(
                q.shape[1] // shift_sa.shape[1], dim=1
            )
            q = self.norm1(q) * (1 + scale_sa) + shift_sa
        else:
            q = self.norm1(q)

        if self.seqra:
            q2, attn = self.seqra(q, self.norm_k(c + pe), self.norm_v(c), padding_mask)
            if self.alpha_scale:
                alpha_ca = self.fc_alpha_ca(torch.cat([q2, q], -1))
                q = (1 + alpha_ca) * q2 + q
            else:
                q = q2 + q
            if self.stepra:
                scale_ca = scale_ca.repeat_interleave(
                    q.shape[1] // scale_ca.shape[1], dim=1
                )
                shift_ca = shift_ca.repeat_interleave(
                    q.shape[1] // shift_ca.shape[1], dim=1
                )
                q = self.norm2(q) * (1 + scale_ca) + shift_ca
            else:
                q = self.norm2(q)

        q = q + self.mlp(q)
        if self.stepra:
            scale_mlp = scale_mlp.repeat_interleave(
                q.shape[1] // scale_mlp.shape[1], dim=1
            )
            shift_mlp = shift_mlp.repeat_interleave(
                q.shape[1] // shift_mlp.shape[1], dim=1
            )
            q = self.norm3(q) * (1 + scale_mlp) + shift_mlp
        else:
            q = self.norm3(q)
        return q, attn


class RADT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.stepra = config.stepra
        self.alpha_scale = config.alpha_scale
        self.seqra = config.seqra
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.model_seq_len = config.seq_len
        self.rtg_scale = config.rtg_scale
        self.pe_sinusoid = config.pe_sinusoid

        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, config.n_embd),
            nn.SiLU(),
        )
        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size + 1, config.n_embd)
        )
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd))

        if self.pe_sinusoid:
            self.pos_emb_table = get_sinusoid_encoding_table(
                config.seq_len + 1, config.n_embd
            )
            self.pos_emb = nn.Linear(config.n_embd, config.n_embd)
            self.global_pos_emb_table = get_sinusoid_encoding_table(
                config.max_timestep + 1, config.n_embd
            )
            self.global_pos_emb = nn.Linear(config.n_embd, config.n_embd)
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.seq_len + 1, config.n_embd)
            )
            self.global_pos_emb = nn.Parameter(
                torch.zeros(1, config.max_timestep + 1, config.n_embd)
            )

        self.q_len = 2
        self.kv_len = 1

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    dim=config.n_embd,
                    num_heads=config.n_head,
                    drop=config.resid_pdrop,
                    attn_drop=config.attn_pdrop,
                    seq_len=config.seq_len,
                    q_len=self.q_len,
                    kv_len=self.kv_len,
                    stepra=config.stepra,
                    alpha_scale=config.alpha_scale,
                    seqra=config.seqra,
                    radt_proj=config.radt_proj,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.input_drop = nn.Dropout(config.embd_pdrop)

        action_emb_dim = config.n_embd
        self.action_head = nn.Linear(action_emb_dim, config.vocab_size)
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        self._init_weights()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm) and module.weight is not None:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_basic_init)
        for block in self.decoder:
            if self.alpha_scale and self.seqra:
                nn.init.constant_(block.fc_alpha_ca.weight, 0)
                nn.init.constant_(block.fc_alpha_ca.bias, 0)
            if self.stepra:
                nn.init.constant_(block.modulation[-1].weight, 0)
                nn.init.constant_(block.modulation[-1].bias, 0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if not self.pe_sinusoid:
            no_decay.add("pos_emb")
            no_decay.add("global_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        assert states.ndim == 3 and states.shape[2] == 4 * 84 * 84, (
            "states: (B, T, 4*84*84) expected"
        )
        assert actions.ndim == 3 and actions.shape[2] == 1, (
            "actions: (B, T, 1) expected"
        )
        if targets is not None:
            assert targets.shape[:2] == states.shape[:2], "targets size != states"
        if rtgs is not None:
            assert rtgs.shape[:2] == states.shape[:2], "rtgs size != states"
        if timesteps is not None:
            assert timesteps.shape[0] == states.shape[0], "timesteps size !== states"

        batch_size, seq_len = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        )  # (batch * block_size, n_embd)
        state_embeddings = state_embeddings.reshape(
            states.shape[0], states.shape[1], self.config.n_embd
        )  # (batch, block_size, n_embd)
        rtg_embeddings = self.ret_emb(rtgs.type(torch.float32) * self.rtg_scale)
        if self.training:
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1)
            )  # (batch, block_size, n_embd)
        else:
            if actions.shape[1] == 0:
                actions = (
                    torch.tensor([self.vocab_size], dtype=torch.long)
                    .to(state_embeddings.device)
                    .unsqueeze(1)
                    .unsqueeze(0)
                )
            else:
                if actions.shape[1] == self.model_seq_len:
                    actions = actions[:, :-1, :]
                actions = torch.cat(
                    [
                        actions,
                        torch.tensor([self.vocab_size], dtype=torch.long)
                        .to(state_embeddings.device)
                        .unsqueeze(1)
                        .unsqueeze(0),
                    ],
                    dim=1,
                )
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1)
            )

        if self.pe_sinusoid:
            all_global_pos_emb = self.global_pos_emb_table.to(timesteps.device)[
                0, timesteps.squeeze()
            ].reshape(batch_size, 1, -1)
            all_global_pos_emb = self.global_pos_emb(all_global_pos_emb)
            local_pos_emb = self.pos_emb_table.to(timesteps.device)[:, :seq_len, :]
            local_pos_emb = self.pos_emb(local_pos_emb)
            local_pos_emb = local_pos_emb.repeat(batch_size, 1, 1)
            position_emb = all_global_pos_emb + local_pos_emb
        else:
            all_global_pos_emb = torch.repeat_interleave(
                self.global_pos_emb, batch_size, dim=0
            )  # batch_size, traj_length, n_embd
            position_emb = (
                torch.gather(
                    all_global_pos_emb,
                    1,
                    torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1),
                )
                + self.pos_emb[:, :seq_len, :]
            )

        query = [state_embeddings + position_emb, action_embeddings + position_emb]
        query = torch.stack(query, dim=1).permute(0, 2, 1, 3)
        query = query.reshape(batch_size, self.q_len * seq_len, self.config.n_embd)

        query = self.input_drop(query)
        condition = rtg_embeddings

        for decoder_block in self.decoder:
            query, _ = decoder_block(query, condition, position_emb, padding_mask=None)

        out = query.reshape(batch_size, seq_len, self.q_len, self.config.n_embd)
        out = out.permute(0, 2, 1, 3)

        state_feat = out[:, 0]

        # get predictions
        logits = self.action_head(state_feat)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
        return logits, loss
