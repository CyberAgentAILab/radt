import torch
import torch.nn as nn

import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""


logger = logging.get_logger(__name__)


class Convolution(nn.Module):
    def __init__(self, config, hidden_size, block_index):
        super().__init__()
        self.window_size = config.window_size
        self.remove_act_embs = config.remove_act_embs

        self.rtg_conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.window_size,
            groups=hidden_size,
        )
        self.obs_conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.window_size,
            groups=hidden_size,
        )
        if not self.remove_act_embs:
            self.act_conv1d = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=self.window_size,
                groups=hidden_size,
            )

    def forward(self, x):
        window_size = self.window_size

        padded_tensor = torch.nn.functional.pad(
            x, (0, 0, window_size - 1, 0)
        ).transpose(1, 2)

        if not self.remove_act_embs:
            rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
            obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
            act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

            conv_tensor = torch.cat(
                (
                    rtg_conv_tensor.unsqueeze(3),
                    obs_conv_tensor.unsqueeze(3),
                    act_conv_tensor.unsqueeze(3),
                ),
                dim=3,
            )
            conv_tensor = conv_tensor.reshape(
                conv_tensor.shape[0], conv_tensor.shape[1], -1
            )

        else:
            rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::2]
            obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::2]

            conv_tensor = torch.cat(
                (rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3)), dim=3
            )
            conv_tensor = conv_tensor.reshape(
                conv_tensor.shape[0], conv_tensor.shape[1], -1
            )

        conv_tensor = conv_tensor.transpose(1, 2).to("cuda")

        return conv_tensor


class Block(nn.Module):
    def __init__(self, config, index, scale=False):
        super().__init__()
        hidden_size = config.n_embd

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.conv = Convolution(config, hidden_size, index)

        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(config.drop_p),
        )

        self.index = index

    def forward(
        self,
        hidden_states,
    ):
        conv_output = self.conv(self.ln_1(hidden_states))
        hidden_states = conv_output + hidden_states

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))

        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(config, index, scale=True) for index in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class DecisionConvFormer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        remove_act_embs=False,
        **kwargs,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            remove_act_embs=remove_act_embs,
            **kwargs,
        )
        self.env_name = config.env_name

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.remove_act_embs = remove_act_embs

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        assert states.ndim == 3, "states: (B, T, D) expected"
        assert actions.ndim == 3, "actions: (B, T, D) expected"
        assert returns_to_go.shape[:2] == states.shape[:2], (
            "returns_to_go size != states"
        )
        assert timesteps.shape[:2] == states.shape[:2], "timesteps size != states"

        batch_size, seq_length = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        if not self.remove_act_embs:
            action_embeddings = self.embed_action(actions) + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.remove_act_embs:
            num_token_type = 2
            stacked_inputs = (
                torch.stack((returns_embeddings, state_embeddings), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, num_token_type * seq_length, self.hidden_size)
            )
            stacked_inputs = self.embed_ln(stacked_inputs)
        else:
            num_token_type = 3
            stacked_inputs = (
                torch.stack(
                    (returns_embeddings, state_embeddings, action_embeddings), dim=1
                )
                .permute(0, 2, 1, 3)
                .reshape(batch_size, num_token_type * seq_length, self.hidden_size)
            )
            stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer(inputs_embeds=stacked_inputs)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(
            0, 2, 1, 3
        )

        state_reps = x[:, 1]

        action_preds = self.predict_action(
            state_reps
        )  # predict next action given state

        return action_preds

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

        states = torch.cat(
            [
                torch.zeros(
                    (
                        states.shape[0],
                        self.max_length - states.shape[1],
                        self.state_dim,
                    ),
                    device=states.device,
                ),
                states,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        actions = torch.cat(
            [
                torch.zeros(
                    (
                        actions.shape[0],
                        self.max_length - actions.shape[1],
                        self.act_dim,
                    ),
                    device=actions.device,
                ),
                actions,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [
                torch.zeros(
                    (
                        returns_to_go.shape[0],
                        self.max_length - returns_to_go.shape[1],
                        1,
                    ),
                    device=returns_to_go.device,
                ),
                returns_to_go,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        timesteps = torch.cat(
            [
                torch.zeros(
                    (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                    device=timesteps.device,
                ),
                timesteps,
            ],
            dim=1,
        ).to(dtype=torch.long)

        action_preds = self.forward(states, actions, returns_to_go, timesteps)

        return action_preds[0, -1], None
