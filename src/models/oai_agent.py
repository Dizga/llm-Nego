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
        if os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model

    def prompt(self, contexts: List[dict]) -> str:
        """
        Generates a response from the OpenAI model based on the provided contexts.

        Args:
            contexts (List[dict]): The contexts for generation.

        Returns:
            str: The generated response from the model.
        """
        if not contexts:
            return ""

        # Assuming contexts is a list of dictionaries with a 'content' key
        prompt_text = " ".join(context["content"] for context in contexts)

        # Call the OpenAI API to generate a response
        response = self.client.Completion.create(
            model=self.model,
            prompt=prompt_text,
            max_tokens=150  # Adjust as needed
        )

        # Extract and return the generated text
        return response.choices[0].text.strip()

    def set_adapter(self, name: str) -> None:
        """
        Dummy method for setting an adapter. Does nothing.
        
        Args:
            name (str): The name of the adapter to switch to.
        """
        pass

    
