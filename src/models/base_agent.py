class BaseAgent:
    def __init__(self):
        self.history = []

    def prompt(self, message: str, is_error=False, is_new_round=False):
        """
        Adds a user message to the conversation history and generates a response.

        Args:
            message (str): The user message to be added to the conversation history.
            is_error (bool): Whether the message is an error.
            is_new_round (bool): Whether the message starts a new round of conversation.

        Returns:
            str: The generated response from the model.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def add_message(self, role, message, is_error=False, is_new_round=False):
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            message (str): The message content.
            is_error (bool): Whether the message is an error.
            is_new_round (bool): Whether the message starts a new round of conversation.
        """
        if is_error:
            # The last assistant message was an error.
            self.history[-1]["is_error"] = True
        self.history.append({"role": role, "content": message, "is_error": is_error, "is_new_round": is_new_round})

    def reset_messages(self):
        """
        Resets the conversation history.
        """
        self.history = []
