import json
import pandas as pd
  
def add_run_to_store(player_1_type, p2_type, player_1_convo, p2_convo, player_1_reward, p2_reward, player_1_behavior, p2_behavior, propositions, state_type):
    runs_df = load_dataframe_from_hdf()
    run_data = {
        "player_1 type": player_1_type,
        "P2 type": p2_type,
        "player_1 Convo": player_1_convo,
        "P2 Convo": p2_convo,
        "player_1 reward": player_1_reward,
        "P2 reward": p2_reward,
        "player_1 behavior": player_1_behavior.value,
        "P2 behavior": p2_behavior.value,
        "Propositions": propositions,
        "State type": state_type,
        "Perfect info": True,
        "Version": 0.1
    }
    # runs_df = runs_df.insert(run_data, ignore_index=True)
    run_data = pd.DataFrame([run_data])
    runs_df = pd.concat([runs_df, run_data], ignore_index=True)
    save_dataframe_to_hdf(runs_df)

def save_dataframe_to_hdf(runs_df, filename="data/negotiation_runs.h5"):
    runs_df.to_hdf(filename, key='df', mode='w')

# Function to load the DataFrame from an HDF5 file
def load_dataframe_from_hdf(filename="data/negotiation_runs.h5"):
    return pd.read_hdf(filename, key='df')