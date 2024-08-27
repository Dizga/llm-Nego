from openai import OpenAI

from agents.base_agent import BaseAgent

class OaiAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini") -> None:
        super().__init__()
        self.openai = OpenAI()
        self.model= model
        
        
    def prompt(self, message, is_error = False, is_new_round = False):

        self.add_message("user", message, is_error=is_error, is_new_round=is_new_round)

        response = self.openai.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=self.history,
        )

        response = response.choices[0].message.content

        self.add_message('assistant', response)
        return response