import json
import numpy as np
import hydra
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import sys
import os

from experiments.dond_ppo_run_train_cycle import dond_ppo_run_train_cycle
from experiments.bc_dond import run_bc_dond

@hydra.main(config_path="../conf", config_name="semicomp")
def main(cfg):
    if os.path.exists('conf/local.yaml'):
        local_cfg = OmegaConf.load('conf/local.yaml')
        cfg = OmegaConf.merge(cfg, local_cfg)
    dond_ppo_run_train_cycle(cfg)
if __name__ == "__main__": main()