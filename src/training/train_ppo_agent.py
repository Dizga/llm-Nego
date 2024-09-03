
import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
# local imports
from environments.dond_game import DondGame
from agents.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent
from training.extract_dond_ppo_dataset import extract_hf_ppo_dataset

def delete_tensor_list(list):
    for el in list: 
        del el

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def train_agent_ppo(
                    agent,
                    folder_path,
                    ppo_trainer_args,
                    nb_epochs,
                    ):

    # Extract training dataset from folder raw data
    queries, responses, scores = extract_hf_ppo_dataset(folder_path, player_0=True)
    queries_player_1, responses_player_1, scores_player_1 = extract_hf_ppo_dataset(folder_path, player_0=False)
    queries = queries + queries_player_1
    responses = responses + responses_player_1
    scores = scores + scores_player_1

    ds = len(queries)
    bs = ppo_trainer_args.batch_size
    nb_batches = ds // bs

    # Get model checkpoint directory
    path = os.path.join(folder_path, 'lora_checkpoints')
    os.makedirs(path, exist_ok=True)

    # Initiate training 
    agent.init_ppo_trainer(os.path.join(folder_path, 'tensorboard'), ppo_trainer_args)
    for _ in range(nb_epochs):
        for b in range(nb_batches):
            beg, end = (b*bs, (b+1)*bs)
            batch_queries, batch_responses, batch_scores = queries[beg:end], responses[beg:end], scores[beg:end]
            stats = agent.train_ppo_json(
                                queries=batch_queries, 
                                responses=batch_responses, 
                                scores=batch_scores, 
                                )
    # Ensure garbage collection is performed
    delete_tensor_list(queries)
    delete_tensor_list(responses)
    delete_tensor_list(scores)
    torch.cuda.empty_cache()

    return stats
        
