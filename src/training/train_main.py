from utils.get_conversations import get_conversations
from training.rl_convs_processing import paths_to_rl_data
from training.ppo_training import ppo_train

def train_main(
        hf_model, 
        paths,
        train_func, 
        train_func_args,
    ):
    globals()[train_func](hf_model, paths, **train_func_args)

def train_ppo_main(
        hf_model,
        paths,
        train_ppo_args,
    ):
    contexts_list, returns_list = paths_to_rl_data(hf_model.tokenizer, paths)
    ppo_train(hf_model.model, hf_model.model, contexts_list, returns_list, **train_ppo_args)
