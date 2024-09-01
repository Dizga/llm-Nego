<<<<<<< HEAD
from typing import Any, List, Dict
from openai import OpenAI
import os


class OaiAgent:
    """
    OaiAgent is an agent that utilizes the OpenAI API for generating responses in a conversational manner.
    It supports prompting the model and managing conversation history.
    """

    def __init__(
        self,
        name: str = "openai_agent",
        api_key: str = "",
        model: str = "gpt-3.5-turbo"  # default OpenAI model
    ) -> None:
        """
        Initializes the OaiAgent.

        Args:
            name (str): The name of the agent.
            api_key (str): The API key for accessing OpenAI's API.
            model (str): The model to be used, specified by the model name (default is 'gpt-3.5-turbo').
            out_folder (str): The output folder for saving conversation history and logs.
        """
        self.name = name
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.history = []


    def _format_messages(self, messages: List[dict]) -> str:
        """
        Formats a list of messages into a single string suitable for display or logging.

        Args:
            messages (List[dict]): List of messages with 'role' and 'content'.

        Returns:
            str: Formatted conversation string.
        """
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted += f"{role}: {content}\n"
        return formatted.strip()

    def add_message(self, role: str, content: str, is_error: bool = False, is_new_round: bool = False) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The message content.
            is_error (bool): Indicates if the message is an error message.
            is_new_round (bool): Indicates if the message starts a new conversation round.
        """
        self.history.append({
            "role": role,
            "content": content,
            "is_error": is_error,
            "is_new_round": is_new_round
        })

    def reset_messages(self) -> None:
        """
        Resets the conversation history.
        """
        self.history = []

    def prompt(self, message: str, is_error: bool = False, is_new_round: bool = False) -> str:
        """
        Sends a user message to the conversation history and generates a response using OpenAI's API.

        Args:
            message (str): The user message to be added to the conversation history.
            is_error (bool): Indicates if the user message is an error message.
            is_new_round (bool): Indicates if the message starts a new conversation round.

        Returns:
            str: The generated response from the model.
        """
        # Add user message to history
        self.add_message(role="user", content=message, is_error=is_error, is_new_round=is_new_round)

        # Generate a response using OpenAI's API
        response = self.client.chat.completions.create(
=======
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
>>>>>>> origin/main
            model=self.model,
            max_tokens=1000,
            messages=self.history,
        )

<<<<<<< HEAD
        # Extract the assistant's response from the response object
        response_text = response.choices[0].message.content

        # Add assistant response to history
        self.add_message(role="assistant", content=response_text)

        return response_text

    def save_history(self, file_path: str) -> None:
        """
        Saves the conversation history to a file in the specified output folder.

        Args:
            file_path (str): The name of the file to save the history to.
        """
        full_path = os.path.join(self.out_folder, file_path)
        with open(full_path, "w") as f:
            for message in self.history:
                f.write(f"{message['role']}: {message['content']}\n")
        print(f"Conversation history saved to {full_path}")

    def load_history(self, file_path: str) -> None:
        """
        Loads the conversation history from a file.

        Args:
            file_path (str): The path to the file containing the conversation history.
        """
        full_path = os.path.join(self.out_folder, file_path)
        with open(full_path, "r") as f:
            self.history = []
            for line in f:
                role, content = line.split(": ", 1)
                self.add_message(role=role, content=content.strip())
        print(f"Conversation history loaded from {full_path}")

    def train(self, train_data_path: str) -> None:
        """
        This is a placeholder method for training the model using OpenAI's API. Currently not implemented.
        
        Args:
            train_data_path (str): The path to the training data file in JSONL format.
        """
        # Placeholder: OpenAI's API does not support direct model fine-tuning via code in this way.
        # You may need to use a different approach or service to fine-tune models.
        print("Training is not implemented for OpenAI API in this agent.")
=======
        response = response.choices[0].message.content

        self.add_message('assistant', response)
        return response
>>>>>>> origin/main
