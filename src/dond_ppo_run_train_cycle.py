import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from utils.dond_logger import DondLogger
from utils.dond_iteration_runner import DondIterationRunner

from environments.dond_game import DondGame
from environments.dond_instructor import DondInstructor
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent
from utils.get_dond_instructor import get_agent
from utils.statistics import *
from utils.train_ppo_agent import train_agent_ppo


def dond_ppo_run_train_cycle(cfg): 
    
    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    logger = DondLogger(output_directory)

    dond_game = DondGame(**cfg.game)

    player_0 = get_agent(dond_game, **cfg.p0)
    player_1 = get_agent(dond_game, **cfg.p1)

    if cfg.p1.agent_args.model_name == "shared": 
        player_1.agent.model = player_0.agent.model

    iteration_runner = DondIterationRunner(
        cfg.playing.games_per_iteration, 
        game=dond_game,
        instructor_0=player_0,
        instructor_1=player_1,
        logger=logger
    )

    # Start training iterations
    for _ in range(cfg.playing.nb_iterations):
        logger.log_info(f"Started playing {cfg.playing.games_per_iteration} games.")
        iteration_runner.run_iteration()
        logger.log_info(f"Completed the {cfg.playing.games_per_iteration} games.")

        log_itr_stats(logger.it_folder)

        logger.log_info("Started PPO training.")
        train_agent_ppo(agent=player_0.agent, 
                        ppo_trainer_args=cfg.training.ppo_trainer_args, 
                        folder_path=logger.it_folder, 
                        nb_epochs=cfg.training.nb_epochs,
                        logger=logger)
        logger.log_info("Ended PPO training.")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    if os.path.exists('conf/local.yaml'):
        local_cfg = OmegaConf.load('conf/local.yaml')
        cfg = OmegaConf.merge(cfg, local_cfg)
    dond_ppo_run_train_cycle(cfg)
if __name__ == "__main__": main()
