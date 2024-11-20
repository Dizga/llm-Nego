import json
import os

def get_conversations(folder_path: str):
    conversations = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r') as f:
            conversations.append(json.load(f))
    return conversations