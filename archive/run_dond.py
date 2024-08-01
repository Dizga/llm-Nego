import json
import numpy as np
from logger import Logger
from DoND import DoND
from agents import NegoAgent
import hydra
from datetime import datetime
import os
from instructor import DoNDInstructor
from hydra.core.hydra_config import HydraConfig

class TwoPlayerNegotiationTrainer:
    def __init__(self, iterations_per_run, games_per_iteration, game, player_0, player_1, logger):
        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.instructor_0 = player_0
        self.instructor_1 = player_1
        self.logger = logger

    def train_agents(self):
        pass

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            self.logger.new_iteration()
            for _ in range(self.games_per_iteration):
                self.run_game()
            #self.train_agents()

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

        

class DoNDTrainer(TwoPlayerNegotiationTrainer):
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

@hydra.main(config_path="../conf", config_name="config")
def run_dond(cfg):

    # Make output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    logger = Logger(output_directory)
    game = DoND(max_turns=cfg.game.max_turns)

    agent_0 = NegoAgent(
        name="agent_0",
        device=cfg.device,
        model=cfg.p0.model,
        tokenizer=cfg.p0.tokenizer,
    )
    player_0 = DoNDInstructor(
        game_intro_file=cfg.p0.game_intro_file,
        chain_of_thought_file=cfg.p0.chain_of_thought,
        proposal_file=cfg.p0.proposal_file,
        dond_game=game,
        dond_player=agent_0,
        player_type="p0"
    )

    agent_1 = NegoAgent(
        name="agent_1",
        device=cfg.device,
        model=cfg.p1.model,
        tokenizer=cfg.p1.tokenizer,
    )
    player_1 = DoNDInstructor(
        game_intro_file=cfg.p1.game_intro_file,
        chain_of_thought_file=cfg.p1.chain_of_thought,
        proposal_file=cfg.p0.proposal_file,
        dond_game=game,
        dond_player=agent_1,
        player_type="p1"
    )

    trainer = DoNDTrainer(
        iterations_per_run=cfg.run.nb_iterations,
        games_per_iteration=cfg.run.games_per_iteration,
        game=game,
        player_0=player_0,
        player_1=player_1,
        logger=logger
    )

    trainer.run_iterations()

if __name__ == "__main__":
    run_dond()
