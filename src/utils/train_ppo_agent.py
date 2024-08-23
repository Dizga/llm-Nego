
import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from utils.dond_logger import DondLogger
from environments.dond_game import DondGame
from environments.dond_instructor import DondInstructor
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
                    logger
                    ):

    # Extract training dataset from folder raw data
    queries, responses, scores = extract_hf_ppo_dataset(folder_path, p0=True)
    queries_p1, responses_p1, scores_p1 = extract_hf_ppo_dataset(folder_path, p0=False)
    queries = queries + queries_p1
    responses = responses + responses_p1
    scores = scores + scores_p1

    # Initiate training 
    for _ in range(nb_epochs):
        bs = ppo_trainer_args.batch_size
        agent.init_ppo_trainer(batch_size=bs)

        for xb in zip(batch(queries, bs), batch(responses, bs), batch(scores, bs)):
            b_queries, b_responses, b_scores = xb
            if len(b_queries) != bs: break
            agent.train_ppo_json(queries=b_queries, responses=b_responses, scores=b_scores)
