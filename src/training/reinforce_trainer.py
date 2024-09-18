from trl import PPOTrainer
import torch
from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)

# see https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1164

# Rewrite the PPO trainer such that the advantages can be passed on manually (as the scores)

class ReinforceTrainer(PPOTrainer):
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
    

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Calculate policy and value losses for REINFORCE by modifying the PPO loss function.

        Args:
            old_logprobs (torch.FloatTensor):
                Log probabilities from the old policy (ignored in REINFORCE).
            values (torch.FloatTensor):
                Values from the old policy (used if employing a baseline).
            logits (torch.FloatTensor):
                Logits from the current policy (used for entropy computation).
            vpreds (torch.FloatTensor):
                Values from the current policy.
            logprobs (torch.FloatTensor):
                Log probabilities from the current policy.
            mask (torch.LongTensor):
                Mask tensor to handle variable sequence lengths.
            advantages (torch.FloatTensor):
                Advantages computed from rewards and baseline values.
            returns (torch.FloatTensor):
                Returns computed from rewards.

        Returns:
            Tuple of policy loss, value loss, and a dictionary of training statistics.
        """

        # OLD CODE
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:

            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )

        # OUR NEW CODE
        pg_loss = -masked_mean(advantages * logprobs, mask)
        vf_loss = 0.5 * masked_mean((vpreds - returns) ** 2, mask)


        return pg_loss, 0.0, flatten_dict(stats)
