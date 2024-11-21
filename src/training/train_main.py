from utils.get_conversations import get_conversations
from training.rl_convs_processing import paths_to_rl_data
from training.ppo_training import ppo_train
from training.reinforce_training import reinforce_train

def train_main(
        hf_model, 
        paths,
        train_func, 
        train_func_args,
        output_path=None,
    ):
    globals()[train_func](hf_model, paths, **train_func_args, output_path=output_path)
    hf_model.export_current_adapter()


def train_ppo_main(
        hf_model,
        paths,
        train_ppo_args={},
        output_path=None,
    ):
    hf_model.train()
    contexts_list, returns_list, output_masks_list = paths_to_rl_data(hf_model.tokenizer, paths)
    ppo_train(model=hf_model.hf_model, 
              ref_model=hf_model.hf_model, 
              contexts_list=contexts_list, 
              returns_list=returns_list, 
              output_masks_list=output_masks_list,
              **train_ppo_args)

def train_reinforce_main(
        hf_model,
        paths,
        train_reinforce_args={},
        output_path=None,
    ):
    hf_model.train()
    contexts_list, returns_list, output_masks_list = paths_to_rl_data(hf_model.tokenizer, paths)
    reinforce_train(model=hf_model.hf_model, 
                    contexts_list=contexts_list, 
                    returns_list=returns_list, 
                    output_masks_list=output_masks_list, 
                    **train_reinforce_args, 
                    output_path=output_path, 
                    tokenizer=hf_model.tokenizer)
