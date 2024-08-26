from typing import Any
import torch
import os
import openai

from agents.hf_agent import HfAgent

class OaiAgent(HfAgent):
    
    def __init__(self,
                 name="openai_agent",
                 api_key="",
                 out_folder="/",
                 model="gpt-3.5-turbo",  # default OpenAI model
                 ) -> None:
        """
        Initializes the OpenAINegoAgent.

        Args:
            name (str): The name of the agent.
            api_key (str): The API key for accessing OpenAI's API.
            out_folder (str): The output folder for saving models and logs.
            model (str): The model to be used, specified by the model name (default is 'gpt-3.5-turbo').
        """
        super().__init__(name=name, out_folder=out_folder)
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model = model

    def train(self, train_data_path):
        """
        Trains the model using OpenAI's API.

        Args:
            train_data_path (str): The path to the training data file in JSONL format.
        """
        # TODO: complete
        openai.File.create(
            file=open(train_data_path, "rb"),
            purpose="fine-tune"
        )
        openai.FineTune.create(
            training_file=train_data_path, 
            model=self.model
        )

    def prompt(self, message: str):
        """
        Adds a user message to the conversation history and generates a response using OpenAI's API.

        Args:
            message (str): The user message to be added to the conversation history.

        Returns:
            str: The generated response from the model.
        """
        user_msg = message
        self.add_message(role="user", message=user_msg)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.history,
        )
        response_text = response['choices'][0]['message']['content']
        self.add_message(role="assistant", message=response_text)
        return response_text