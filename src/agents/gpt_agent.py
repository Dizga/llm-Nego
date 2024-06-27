from typing import Any
from openai import OpenAI

class GPTAgent():
    def __init__(self, name) -> None:
        self.messages = []
        self.openai = OpenAI()
        self.name = name
        
        
    def __call__(self, add_to_history = True, *args: Any, **kwds: Any) -> Any:
        response = self.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=500,
            messages=self.messages,
        )

        response = response.choices[0].message.content

        if add_to_history : self.add_message('assistant', response)
        return response
    
    def add_message(self, role, message):
        self.messages.append({"role": role, "content": message})

    def reset_messages(self):
        self.messages = []

    def add_system_message(self, message):
        self.add_message("system", message)