import json
import numpy as np
import hydra
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import sys
import os

from experiments.dond_ppo_run_train_cycle import dond_ppo_run_train_cycle
#from experiments.ultimatum_run import ultimatum
from experiments.training_tester import training_tester
from experiments.simple_test_2 import simple_test_2
from experiments.simple_test import simple_test
from experiments.arithmetic_test import arithmetic_test

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    dond_ppo_run_train_cycle(cfg)
    #simple_test()
    #arithmetic_test()

if __name__ == "__main__": main()

