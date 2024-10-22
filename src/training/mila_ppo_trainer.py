
import torch
import accelerate


def weighted_grad_step(model, tokenizer, messages):
    # Messages with scores

    for conv in convs:
        for message in conv:
            if message["role"] == "user":
                user_input = message["content"]
            else:
                model_output = message["content"]
            # Todo: add to context