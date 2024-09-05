# local imports
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent
from utils.inherit_args import inherit_args

def setup_players(cfg, player_type):
    agents = []
    inherit_args(cfg.player_0, cfg.player_1, "same_as_player_0")
    for player_cfg in [cfg.player_0, cfg.player_1]:
        if player_cfg.type == "hf":
            agent = HfAgent(**player_cfg.agent_args)
        elif player_cfg.type == "dummy_hf":
            agent = DummyHfAgent(**player_cfg.agent_args)
        elif player_cfg.type == "oai":
            agent = OaiAgent(**player_cfg.agent_args)
        agents.append(agent)
    
    if cfg.player_1.agent_args.inherit_model and cfg.player_0.type != "oai": 
        agents[1].model = agents[0].model

    players = [
        player_type(**cfg.player_0.player_args, agent=agents[0]),
        player_type(**cfg.player_1.player_args, agent=agents[1])
    ]
    return players
