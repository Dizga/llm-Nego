import torch.nn.functional as F
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def ppo_train( 
        model, 
        ref_model,
        contexts_list,
        returns_list,
        output_masks_list,
        optimizer=None, 
        nb_epochs=1,
        mb_size=1,
        mb_per_step=1,
        learning_rate=1e-3,
        clip_param=0.2, 
        vf_coef=0.5,
        entropy_coef=0.01):
    """
    Perform a single PPO training step.

    Args:
        model (torch.nn.Module): The language model with a value head to be optimized.
        ref_model (torch.nn.Module): Reference model used for KL penalty.
        contexts_list (list of torch.Tensor): List of input contexts, each of shape (S, V).
        returns_list (list of torch.Tensor): List of estimated returns for each time step, each of shape (S,).
        output_masks_list (list of torch.Tensor): List of masks for output tokens, each of shape (S,).
        optimizer (torch.optim.Optimizer, optional): Optimizer for training the model. If None, a default optimizer will be created.
        nb_epochs (int): Number of epochs to train over the dataset.
        mb_size (int): Minibatch size, the number of sequences processed at once.
        mb_per_step (int): Number of minibatches to accumulate gradients over before taking an optimizer step.
        clip_param (float, optional): Clipping parameter epsilon for PPO, default is 0.2.
        vf_coef (float, optional): Coefficient for value loss, default is 0.5.
        entropy_coef (float, optional): Coefficient for entropy bonus, default is 0.01.

    Returns:
        float: The total loss value for the training step.
    """
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    verify_ppo_train_inputs(contexts_list, returns_list, output_masks_list)

    # Initialize the accelerators
    model_accelerator = Accelerator()
    ref_model_accelerator = Accelerator()
    model, optimizer = model_accelerator.prepare(model, optimizer)
    ref_model = ref_model_accelerator.prepare(ref_model)

    model.train()

    for epoch in range(nb_epochs):
        for i in range(0, len(contexts_list), mb_size):
            # Get the minibatch
            context_batch = contexts_list[i:i + mb_size]
            return_batch = returns_list[i:i + mb_size]
            mask_batch = output_masks_list[i:i + mb_size]

            # Pad sequences
            context_batch = pad_sequence(context_batch, batch_first=True).long()
            return_batch = pad_sequence(return_batch, batch_first=True).float()
            mask_batch = pad_sequence(mask_batch, batch_first=True).float()

            # Create attention mask to ignore padding tokens
            attention_mask = (context_batch != 0).long()

            # Move data to the appropriate device
            context_batch = context_batch.to(model_accelerator.device)
            return_batch = return_batch.to(model_accelerator.device)
            mask_batch = mask_batch.to(model_accelerator.device)
            attention_mask = attention_mask.to(model_accelerator.device)

            # Forward pass for the reference model to compute old log probabilities
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=context_batch, attention_mask=attention_mask)
                ref_logits = ref_outputs[0]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                old_log_probs = ref_log_probs.gather(dim=-1, index=context_batch.unsqueeze(-1)).squeeze(-1)

            # Forward pass
            logits, _, values = model(input_ids=context_batch, attention_mask=attention_mask)

            # Compute new log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(dim=-1, index=context_batch.unsqueeze(-1)).squeeze(-1)

            # Apply mask to log probabilities and values
            action_log_probs *= mask_batch
            values *= mask_batch
            return_batch *= mask_batch

            # Compute entropy for entropy bonus
            entropy = - (log_probs * torch.exp(log_probs)).sum(-1).mean()

            # Compute ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(action_log_probs - old_log_probs)

            # Compute advantages
            advantages = return_batch - values

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value function loss
            value_loss = F.mse_loss(values, return_batch)

            # Total loss
            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

            # Accumulate gradients
            model_accelerator.backward(loss)

            # Perform optimizer step every mb_per_step minibatches
            if (i // mb_size + 1) % mb_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

    return loss.item()


def verify_ppo_train_inputs(contexts_list, returns_list, output_masks_list):
    """
    Verify the inputs to the ppo_train function.
    """
    for context, returns, mask in zip(contexts_list, returns_list, output_masks_list):
        assert context.size(0) == returns.size(0) == mask.size(0), (
            f"Context, returns, and mask lengths do not match. "
            f"Context shape: {context.shape}, Returns shape: {returns.shape}, Mask shape: {mask.shape}"
        )
        
