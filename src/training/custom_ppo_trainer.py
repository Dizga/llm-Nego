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
        """
        Compute advantages without using value function approximation.
        Advantages are set equal to the rewards.
        """
        # Ignore values, just use rewards
        values = torch.zeros_like(rewards)
        rewards = rewards * mask
        advantages = rewards
        returns = rewards
        return values, advantages, returns