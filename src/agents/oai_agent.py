from typing import Any
from openai import OpenAI

from agents.hf_agent import HfAgent

class OaiAgent():
    def __init__(self, model="gpt-4o-mini") -> None:
        self.history = []
        self.openai = OpenAI()
        self.model= model
        
        
    def prompt(self, message, is_error = False, is_new_round = False):

        self.add_message("user", message)

        response = self.openai.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=self.history,
        )

        response = response.choices[0].message.content

        self.add_message('assistant', response)
        return response
    
    def add_message(self, role, message):
        self.history.append({"role": role, "content": message})

    def reset_messages(self):
        self.history = []