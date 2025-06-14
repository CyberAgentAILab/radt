import torch

from training.trainer import Trainer


class DecisionTransformerTrainer(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = (
            self.get_batch(self.batch_size)
        )
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds, attentions = self.model.forward(
            states, actions, rtg[:, :-1], timesteps, attention_mask=attention_mask
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        return loss.detach().cpu().item()
