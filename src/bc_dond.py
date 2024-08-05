import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from utils.bc_dond_logger import BcDondLogger
from environments.dond_game import DondGame
from environments.dond_instructor import DondInstructor
from agents.hf_agent import HfAgent
from agents.oai_agent import OaiAgent

class BcDondTrainer:
    def __init__(self, iterations_per_run, games_per_iteration, game, instructor_0, instructor_1, logger):
        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.instructor_0 = instructor_0
        self.instructor_1 = instructor_1
        self.logger = logger

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            self.logger.new_iteration()
            for _ in range(self.games_per_iteration):
                self.run_game()
            self.train_agents()

    def run_game(self):
        self.logger.log_info("Game started.")
        self.logger.new_game()
        self.instructor_0.new_game()
        self.instructor_1.new_game()
        self.game.reset()
        while True:
            if not self.instructor_0.play_move(): break
            if not self.instructor_1.play_move(): break
        self.logger.log_game(self.game.export(), self.instructor_0.get_history(), self.instructor_1.get_history())
        self.logger.log_info("Game completed.")


    def train_agents(self):
        """Train the agents on the last iteration."""
        metrics = self.logger.metrics # Extract dataframe with data for each game
        mean_score_p0 = self.logger.iteration_stats['Mean Score P0'] # Get the mean score of the current iteration
        mean_score_p1 = self.logger.iteration_stats['Mean Score P1'] # Get the mean score of the current iteration
        # Filter games with score better than the mean score
        filtered_p0 = metrics[metrics['p0_score'] >= mean_score_p0]
        filtered_p1 = metrics[metrics['p1_score'] >= mean_score_p1]
        p0_filtered_files = [self.logger.it_folder + '/' + element for element in filtered_p0['p0_file'].tolist()]
        p1_filtered_files = [self.logger.it_folder + '/' + element for element in filtered_p1['p1_file'].tolist()]
        p0_filtered_jsons = [json.load(open(file_path, 'r')) for file_path in p0_filtered_files]
        p1_filtered_jsons = [json.load(open(file_path, 'r')) for file_path in p1_filtered_files]
        self.instructor_0.dond_player.train(p0_filtered_jsons)
        self.instructor_1.dond_player.train(p1_filtered_jsons)

def run_bc_dond(cfg):
    # Run behaviour cloning for the deal-or-no-deal game

    # Make output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    logger = BcDondLogger(output_directory)
    game = DondGame(max_turns=cfg.game.max_turns,
                    setup=cfg.game.setup,
                    setups_file=cfg.game.setups_file
                    )

    # Get the player/instructor 0
    if cfg.p0.type == "hf":
        agent_0 = HfAgent(
            name="agent_0",
            device=cfg.device,
            model=cfg.p0.model,
            tokenizer=cfg.p0.tokenizer,
            out_folder=output_directory + "/checkpoints"
        )
    elif cfg.p0.type == "oai":
        agent_0 = HfAgent(
            name="agent_0",
            device=cfg.device,
            model=cfg.p0.model,
            tokenizer=cfg.p0.tokenizer,
        )
    instructor_0 = DondInstructor(
        game_intro_file=cfg.p0.game_intro_file,
        chain_of_thought_file=cfg.p0.chain_of_thought,
        proposal_file=cfg.p0.proposal_file,
        dond_game=game,
        dond_player=agent_0,
        player_type="p0"
    )

    # Get player/instructor 1
    if cfg.p1.type == "hf":
        agent_1 = HfAgent(
            name="agent_1",
            device=cfg.device,
            model=cfg.p1.model,
            tokenizer=cfg.p1.tokenizer,
            out_folder=output_directory + "/checkpoints"
        )
    elif cfg.p1.type == "oai":
        agent_1 = OaiAgent(
            name="agent_1",
            device=cfg.device,
            model=cfg.p1.model,
            tokenizer=cfg.p1.tokenizer,
        )
    instructor_1 = DondInstructor(
        game_intro_file=cfg.p1.game_intro_file,
        chain_of_thought_file=cfg.p1.chain_of_thought,
        proposal_file=cfg.p0.proposal_file,
        dond_game=game,
        dond_player=agent_1,
        player_type="p1"
    )

    trainer = BcDondTrainer(
        iterations_per_run=cfg.run.nb_iterations,
        games_per_iteration=cfg.run.games_per_iteration,
        game=game,
        instructor_0=instructor_0,
        instructor_1=instructor_1,
        logger=logger
    )
    trainer.run_iterations()



@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    if os.path.exists('conf/local.yaml'):
        local_cfg = OmegaConf.load('conf/local.yaml')
        cfg = OmegaConf.merge(cfg, local_cfg)
    run_bc_dond(cfg)
if __name__ == "__main__": main()
