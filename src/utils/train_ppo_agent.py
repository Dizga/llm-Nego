
import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
# local imports
from utils.dond_logger import DondLogger
from environments.dond_game import DondGame
from environments.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent
from utils.extract_dond_ppo_dataset import extract_hf_ppo_dataset


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

    bs = len(queries) - (len(queries) % ppo_trainer_args.mini_batch_size)
    queries, responses, scores = queries[:bs], responses[:bs], scores[:bs] 
    ppo_trainer_args.batch_size = bs

    # Get model checkpoint directory
    path = os.path.join(folder_path, 'lora_checkpoints')
    os.makedirs(path, exist_ok=True)

    # Initiate training 
    for _ in range(nb_epochs):
        agent.init_ppo_trainer(ppo_trainer_args)
        stats = agent.train_ppo_json(
                             directory=folder_path,
                             queries=queries, 
                             responses=responses, 
                             scores=scores, 
                            )
        
    return stats
        
