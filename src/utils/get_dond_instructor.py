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


def get_dond_player(
        dond_game,
        agent_args,
        player_args
    ):
    if type == "hf": agent_0 = HfAgent(**agent_args)
    if type == "dummy_hf": agent_0 = DummyHfAgent(**agent_args)
    elif type == "oai": agent_0 = HfAgent(**agent_args)
    instructor = DondInstructor(
        **player_args, 
        dond_game=dond_game,
        dond_player=agent_0, 
    )
    return instructor()
