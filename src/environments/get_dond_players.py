import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from agents.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent


def get_dond_players(
    dond_game,
    player_0_args,
    player_1_args,
):
    # Create agent for player 0
    if player_0_args.type == "hf":
        agent_0 = HfAgent(**player_0_args.agent_args)
        agent_0._initialize_model(**player_0_args.model_args)
    elif player_0_args.type == "dummy_hf":
        agent_0 = DummyHfAgent(**player_0_args.agent_args)
    elif player_0_args.type == "oai":
        agent_0 = OaiAgent(**player_0_args.agent_args)
    else:
        raise ValueError(f"Unknown agent type: {player_0_args.type}")

    # Create agent for player 1
    if player_1_args.type == "hf":
        agent_1 = HfAgent(**player_1_args.agent_args)
        if player_1_args.inherit_model:
            agent_1.model = agent_0.model
        else: 
            agent_1._initialize_model(**player_1_args.model_args)
    elif player_1_args.type == "dummy_hf":
        agent_1 = DummyHfAgent(**player_1_args.agent_args)
    elif player_1_args.type == "oai":
        agent_1 = OaiAgent(**player_1_args.agent_args)
    else:
        raise ValueError(f"Unknown agent type: {player_1_args.type}")


    # Create players
    player_0 = DondPlayer(
        dond_game,
        **player_0_args.player_args, 
        agent=agent_0, 
    )
    
    player_1 = DondPlayer(
        dond_game,
        **player_1_args.player_args, 
        agent=agent_1, 
    )

    return player_0, player_1

