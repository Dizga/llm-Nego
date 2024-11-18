import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def ppo_train( 
        model, 
        ref_model,
        contexts_list,
        returns_list,
        optimizer, 
        nb_epochs,
        mb_size,
        mb_per_step,
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
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        nb_epochs (int): Number of epochs to train over the dataset.
        mb_size (int): Minibatch size, the number of sequences processed at once.
        mb_per_step (int): Number of minibatches to accumulate gradients over before taking an optimizer step.
        clip_param (float, optional): Clipping parameter epsilon for PPO, default is 0.2.
        vf_coef (float, optional): Coefficient for value loss, default is 0.5.
        entropy_coef (float, optional): Coefficient for entropy bonus, default is 0.01.

    Returns:
        float: The total loss value for the training step.
    """
    verify_ppo_train_inputs(contexts_list, returns_list)

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

            # Pad sequences to create a batch tensor
            input_batch = pad_sequence([c[:-1] for c in context_batch], batch_first=True)
            label_batch = pad_sequence([c[1:] for c in context_batch], batch_first=True)
            return_batch = pad_sequence(return_batch, batch_first=True)

            # Move data to the appropriate device
            input_batch = input_batch.to(model_accelerator.device)
            label_batch = label_batch.to(model_accelerator.device)
            return_batch = return_batch.to(model_accelerator.device)

            # Forward pass for the reference model to compute old log probabilities
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_batch, labels=label_batch)
                ref_logits = ref_outputs[0]  # Unpack the first element of the tuple
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # shape (B, S-1, V)
                old_log_probs = ref_log_probs.gather(dim=-1, index=label_batch.unsqueeze(-1)).squeeze(-1)  # shape (B, S-1)

            # Forward pass
            logits, values = model(input_ids=input_batch, labels=label_batch)

            # Compute new log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # shape (B, S-1, V)
            action_log_probs = log_probs.gather(dim=-1, index=label_batch.unsqueeze(-1)).squeeze(-1)  # shape (B, S-1)

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


def verify_ppo_train_inputs(contexts_list, returns_list):
    """
    Verify the inputs to the ppo_train function.
    """
    for context, returns in zip(contexts_list, returns_list):
        assert context.size(0) == returns.size(0), (
            f"Context and returns lengths do not match. "
            f"Context shape: {context.shape}, Returns shape: {returns.shape}"
        )
        
