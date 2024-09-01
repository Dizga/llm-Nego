import hydra
import os
import os
import logging
import logging.config


# local imports
from utils.dond_iteration_runner import DondIterationRunner
from environments.dond_game import DondGame
from environments.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent
from utils.get_dond_player import get_agents
from utils.train_ppo_agent import train_agent_ppo
from utils.dond_statistics import compute_dond_statistics
from utils.log_gpu_usage import log_gpu_usage


def dond_ppo_run_train_cycle(cfg): 
    
    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)


    dond_game = DondGame(**cfg.game)
    inherit_args(cfg.player_0, cfg.player_1, "same_as_player_0")
    player_0, player_1 = get_agents(dond_game, cfg.player_0, cfg.player_1)

    iteration_runner = DondIterationRunner(
        output_directory,
        cfg.playing.games_per_iteration, 
        game=dond_game,
        player_0=player_0,
        player_1=player_1,
    )


    for _ in range(cfg.playing.nb_iterations):
        
        # Play games
        logging.info(f"Started playing {cfg.playing.games_per_iteration} games.")
        log_gpu_usage()
        iteration_runner.run_iteration()
        logging.info(f"Completed the {cfg.playing.games_per_iteration} games.")
        log_gpu_usage()


        compute_dond_statistics(iteration_runner.it_folder)

        # Train on games played
        logging.info(f"Started {cfg.training.train_type} training.")


        if cfg.training.train_type == "ppo":
            train_agent_ppo(agent=player_0.agent, 
                            ppo_trainer_args=cfg.training.ppo_trainer_args, 
                            folder_path=iteration_runner.it_folder, 
                            nb_epochs=cfg.training.nb_epochs,
                            )
        elif cfg.training.train_type == "bc":
            # TODO
            pass

        if cfg.training.checkpoint_models: # Export LoRA model weights
            player_0.agent.checkpoint_model(iteration_runner.it_folder)

        logging.info(f"Ended {cfg.training.train_type} training.")



