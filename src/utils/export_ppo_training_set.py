import json
import os
import copy
from statistics import mean
import regex as re


def export_ppo_training_set(file_path, queries, responses, scores):
    debug_data = [{"query": q, "response": r, "score": s} 
                    for q, r, s in zip(queries, responses, scores)]
    with open(file_path, 'w') as debug_file:
        json.dump(debug_data, debug_file, indent=4)