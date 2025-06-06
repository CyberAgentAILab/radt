import torch

from training.trainer import Trainer


class StarformerTrainer(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, padding_mask = self.get_batch(
            self.batch_size
        )
        action_target = torch.clone(actions[:, 1:])
        action_input = actions[:, :-1]

        state_preds, action_preds, reward_preds = self.model.forward(
            states, action_input, rtg[:, :-1], timesteps, padding_mask=padding_mask
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[padding_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[padding_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        return loss.detach().cpu().item(), None
