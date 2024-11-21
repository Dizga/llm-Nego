import torch.nn.functional as F
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
import random

def reinforce_train( 
        model, 
        contexts_list,
        returns_list,
        output_masks_list,
        optimizer=None, 
        nb_epochs=1,
        mb_size=1,
        mb_per_step=1,
        learning_rate=1e-5,
        output_path=None,
        tokenizer=None
        ):
    """

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
    model.train()
    if output_path: 
        output_train_data_debug(output_path, 
                                contexts_list, 
                                returns_list, 
                                output_masks_list, 
                                tokenizer)

    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    verify_reinforce_train_inputs(contexts_list, returns_list, output_masks_list)

    # Initialize the accelerators
    model_accelerator = Accelerator()
    model, optimizer = model_accelerator.prepare(model, optimizer)


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

            # Forward pass
            outputs = model(input_ids=context_batch, attention_mask=attention_mask)
            logits = outputs[0]
            # Compute new log probabilities            
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(dim=-1, index=context_batch.unsqueeze(-1)).squeeze(-1)

            # Apply mask to log probabilities and values
            action_log_probs *= (return_batch * mask_batch)
            loss = -action_log_probs.mean()

            # Accumulate gradients
            model_accelerator.backward(loss)

            # Perform optimizer step every mb_per_step minibatches
            if (i // mb_size + 1) % mb_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

    return loss.item()


def verify_reinforce_train_inputs(contexts_list, returns_list, output_masks_list):
    """
    Verify the inputs to the reinforce_train function.
    """
    for context, returns, mask in zip(contexts_list, returns_list, output_masks_list):
        assert context.size(0) == returns.size(0) == mask.size(0), (
            f"Context, returns, and mask lengths do not match. "
            f"Context shape: {context.shape}, Returns shape: {returns.shape}, Mask shape: {mask.shape}"
        )

def output_train_data_debug(path, contexts_list, returns_list, output_masks_list, tokenizer):
    """
    Output the training data for debugging.
    
    Args:
        path (str): The directory path where the output files will be saved.
        contexts_list (list of torch.Tensor): List of input contexts, each of shape (S, V).
        returns_list (list of torch.Tensor): List of estimated returns for each time step, each of shape (S,).
        output_masks_list (list of torch.Tensor): List of masks for output tokens, each of shape (S,).
        tokenizer: Tokenizer to convert token IDs to their written form.
    """
    path = os.path.join(path, "train_debug", str(random.randint(0, 1000)))
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    for idx, (context, returns, mask) in enumerate(zip(contexts_list, returns_list, output_masks_list)):
        # Convert token IDs to written form
        tokens = tokenizer.convert_ids_to_tokens(context.tolist())

        # Prepare the triplets
        triplets = list(zip(tokens, returns.tolist(), mask.tolist()))

        # Define the file path for the current conversation
        file_path = os.path.join(path, f"conversation_{idx}.txt")

        # Write the triplets to the file
        with open(file_path, 'w') as f:
            for token, ret, msk in triplets:
                f.write(f"{token}\t{ret}\t{msk}\n")
        
        