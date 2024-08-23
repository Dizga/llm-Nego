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


class BcDondTrainer:
    def __init__(self, 
                 iterations_per_run, 
                 games_per_iteration, 
                 game: DondGame, 
                 train_type: str,
                 instructor_0: DondInstructor, 
                 instructor_1: DondInstructor, 
                 logger: DondLogger,
                 ):

        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.instructor_0 = instructor_0
        self.instructor_1 = instructor_1
        self.logger = logger
        self.train_type = train_type

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            folder_path = self.logger.new_iteration()
            for _ in range(self.games_per_iteration):
                self.run_game()
            if self.train_type == "ppo":
                self.train_agents_ppo(folder_path)
            else:
                self.train_agents_bc(folder_path)


    def run_game(self):
        self.logger.log_info("Game started.")
        self.logger.new_game()
        instructors = [self.instructor_0, self.instructor_1]
        self.instructor_0.new_game()
        self.instructor_1.new_game()
        game_state = self.game.reset()
        player_id = 0
        while not game_state['game_ended']:
            if game_state['new_round']:
                self.instructor_0.new_round()
                self.instructor_1.new_round()
            is_proposal, content = instructors[player_id].play_move(game_state)
            game_state = self.game.step(content, is_proposal=is_proposal)
            player_id = (player_id + 1) % 2
            
        # while True:
        #     if self.instructor_0.play_move(): break
        #     if self.instructor_1.play_move(): break
        self.logger.log_game(*self.game.export(), 
                             self.instructor_0.get_history(), 
                             self.instructor_1.get_history())
        self.logger.log_info("Game completed.")

    def train_agents_ppo(self, folder_path):

        self.logger.log_info("PPO training started.")
        
        # Train player 0
        self.instructor_0.agent.init_ppo_trainer()
        queries, responses, scores = self.logger.extract_hf_ppo_dataset(folder_path, p0=True)
        self.instructor_0.agent.train_ppo_json(queries, responses, scores)

        # Train player 1
        self.instructor_1.agent.init_ppo_trainer()
        queries, responses, scores = self.logger.extract_hf_ppo_dataset(folder_path, p0=False)
        self.instructor_1.agent.train_ppo_json(queries, responses, scores)

        self.logger.log_info("PPO training ended.")


    def train_agents_bc(self):
        """Train the agents on the last iteration."""
        # TODO: update this function
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
        self.instructor_0.agent.train(p0_filtered_jsons)
        self.instructor_1.agent.train(p1_filtered_jsons)




def run_bc_dond(cfg): # TODO: change name
    
    # Make output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    # Get input and output handler for DonD
    logger = DondLogger(output_directory)

    # Get instance which handles the game
    game = DondGame(**cfg.game)

    # Get the player/instructor 0
    if cfg.players.p0.type == "hf": agent_0 = HfAgent(**cfg.players.p0.agent_args)
    if cfg.players.p0.type == "dummy_hf": agent_0 = DummyHfAgent(**cfg.players.p0.agent_args)
    elif cfg.players.p0.type == "oai": agent_0 = HfAgent(**cfg.players.p0.agent_args)
    instructor_0 = DondInstructor(
        **cfg.players.p0.instructor_args, dond_game=game,
        agent=agent_0, player_type="p0"
    )

    # Get player/instructor 1
    if cfg.players.shared_model: cfg.players.p1.agent_args.model_name = None
    if cfg.players.p1.type == "hf": agent_1 = HfAgent(**cfg.players.p1.agent_args)
    elif cfg.players.p1.type == "dummy_hf": agent_1 = DummyHfAgent(**cfg.players.p1.agent_args)
    elif cfg.players.p1.type == "oai": agent_1 = HfAgent(**cfg.players.p1.agent_args)
    if cfg.players.shared_model: agent_1.model = agent_0.model

    instructor_1 = DondInstructor(
        **cfg.players.p0.instructor_args, dond_game=game,
        agent=agent_1, player_type="p1"
    )

    # Start training
    trainer = BcDondTrainer(
        **cfg.training, game=game,
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
