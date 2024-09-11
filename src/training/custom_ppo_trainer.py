from trl import PPOTrainer
import torch

# see https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1164

# Rewrite the PPO trainer such that the advantages can be passed on manually (as the scores)

class CustomPPOTrainer(PPOTrainer):
    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        values = values * mask
        rewards = rewards * mask
        advantages = rewards
        returns = advantages + values
        return values, advantages, returns