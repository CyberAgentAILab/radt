import torch
import torch.nn as nn
from einops import rearrange

from .layer import get_sinusoid_encoding_table


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

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


# Pre-Norm
class PreNormDecoderBlock(nn.Module):
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

        if self.stepra:
            scale_sa = scale_sa.repeat_interleave(
                q.shape[1] // scale_sa.shape[1], dim=1
            )
            shift_sa = shift_sa.repeat_interleave(
                q.shape[1] // shift_sa.shape[1], dim=1
            )
            q1 = self.norm1(q) * (1 + scale_sa) + shift_sa
        else:
            q1 = self.norm1(q)
        q1, attn = self.attn(q1, q1, q1, padding_mask)
        q = q1 + q

        if self.seqra:
            if self.stepra:
                scale_ca = scale_ca.repeat_interleave(
                    q.shape[1] // scale_ca.shape[1], dim=1
                )
                shift_ca = shift_ca.repeat_interleave(
                    q.shape[1] // shift_ca.shape[1], dim=1
                )
                q2 = self.norm2(q) * (1 + scale_ca) + shift_ca
            else:
                q2 = self.norm2(q)
            q2, attn = self.seqra(q2, self.norm_k(c + pe), self.norm_v(c), padding_mask)
            if self.alpha_scale:
                alpha_ca = self.fc_alpha_ca(torch.cat([q2, q], -1))
                q = (1 + alpha_ca) * q2 + q
            else:
                q = q2 + q

        if self.stepra:
            scale_mlp = scale_mlp.repeat_interleave(
                q.shape[1] // scale_mlp.shape[1], dim=1
            )
            shift_mlp = shift_mlp.repeat_interleave(
                q.shape[1] // shift_mlp.shape[1], dim=1
            )
            q3 = self.norm3(q) * (1 + scale_mlp) + shift_mlp
        else:
            q3 = self.norm3(q)
        q = q + self.mlp(q3)
        return q, attn


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

        q1, self_attn_score = self.attn(q, q, q, padding_mask)
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

        seqra_score = None
        if self.seqra:
            q2, seqra_score = self.seqra(
                q, self.norm_k(c + pe), self.norm_v(c), padding_mask
            )
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
        return q, seqra_score


class RADT(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        seq_len: int = 20,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        use_learnable_pos_emb: bool = False,
        stepra: bool = False,
        alpha_scale: bool = False,
        seqra: bool = False,
        remove_act_embs: bool = False,
        action_tanh: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.max_length = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.episode_len = episode_len
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.stepra = stepra
        self.alpha_scale = alpha_scale
        self.seqra = seqra
        self.remove_act_embs = remove_act_embs

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(act_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)
        if self.use_learnable_pos_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        else:
            self.timestep_table = get_sinusoid_encoding_table(
                episode_len + seq_len, embedding_dim
            )
            self.timestep_emb = nn.Linear(embedding_dim, embedding_dim)

        if self.remove_act_embs:
            self.q_len = 1
        else:
            self.q_len = 2
        self.kv_len = 1

        if stepra:
            self.decoder = nn.ModuleList(
                [
                    DecoderBlock(
                        dim=embedding_dim,
                        num_heads=num_heads,
                        drop=residual_dropout,
                        attn_drop=attention_dropout,
                        seq_len=seq_len,
                        q_len=self.q_len,
                        kv_len=self.kv_len,
                        stepra=self.stepra,
                        alpha_scale=self.alpha_scale,
                        seqra=self.seqra,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.decoder = nn.ModuleList(
                [
                    PreNormDecoderBlock(
                        dim=embedding_dim,
                        num_heads=num_heads,
                        drop=residual_dropout,
                        attn_drop=attention_dropout,
                        seq_len=seq_len,
                        q_len=self.q_len,
                        kv_len=self.kv_len,
                        stepra=self.stepra,
                        alpha_scale=self.alpha_scale,
                        seqra=self.seqra,
                    )
                    for _ in range(num_layers)
                ]
            )

        action_emb_dim = embedding_dim
        if action_tanh:
            self.action_head = nn.Sequential(
                nn.Linear(action_emb_dim, act_dim), nn.Tanh()
            )
        else:
            self.action_head = nn.Linear(action_emb_dim, act_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
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

    def forward(
        self, states, actions, returns_to_go, timesteps, padding_mask=None
    ) -> torch.FloatTensor:
        assert states.ndim == 3, "states: (B, T, D) expected"
        assert actions.ndim == 3, "actions: (B, T, D) expected"
        assert returns_to_go.shape[:2] == states.shape[:2], (
            "returns_to_go size != states"
        )
        assert timesteps.shape[:2] == states.shape[:2], "timesteps size != states"

        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        if self.use_learnable_pos_emb:
            timestep_emb = self.timestep_emb(timesteps)
        else:
            timestep_tab = self.timestep_table.to(timesteps.device)[0, timesteps, :]
            timestep_emb = self.timestep_emb(timestep_tab)

        if actions.shape[1] > seq_len:
            actions = actions[:, :seq_len]

        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        returns_emb = self.return_emb(returns_to_go)

        if self.remove_act_embs:
            query = [state_emb + timestep_emb]
        else:
            query = [state_emb + timestep_emb, act_emb + timestep_emb]
        query = torch.stack(query, dim=1).permute(0, 2, 1, 3)
        query = query.reshape(batch_size, self.q_len * seq_len, self.embedding_dim)

        attn_weights = []
        for decoder_block in self.decoder:
            query, seqra_score = decoder_block(
                query, returns_emb, timestep_emb, padding_mask=padding_mask
            )
            if seqra_score is not None:
                attn_weights.append(seqra_score.detach().cpu().numpy())

        out = query.reshape(batch_size, seq_len, self.q_len, self.embedding_dim)
        out = out.permute(0, 2, 1, 3)

        state_feat = out[:, 0]

        # get predictions
        action_preds = self.action_head(state_feat)

        return action_preds, attn_weights

    def get_action(self, states, actions, returns_to_go, timesteps):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -self.max_length :]
        actions = actions[:, -self.max_length :]
        returns_to_go = returns_to_go[:, -self.max_length :]
        timesteps = timesteps[:, -self.max_length :]

        action_preds, attn_weights = self.forward(
            states, actions, returns_to_go, timesteps
        )

        return action_preds[0, -1], attn_weights
