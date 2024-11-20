Environments should follow a specific interface that allows for consistent interaction. Here are the key components and forms that an environment should take:

## Environment Interface

1. **Initialization (`__init__` method):**
   - Define the environment's state space, action space, and any other necessary parameters.
   - Example:
     ```python
     def __init__(self):
         self.state_space = ...
         self.action_space = ...
         # Initialize other parameters
     ```

2. **Reset Method:**
   - Resets the environment to an initial state and returns the initial observation.
   - Example:
     ```python
     def reset(self):
         # Reset the state of the environment to an initial state
         return initial_observation
     ```

3. **Act Method:**
   - ...
   - Example:
     ```python
     def act(self, state, info, llm_output):
         return send_to_game, action
     ```

3. **Export Method:**
   - Exports every relevant information to a file. Includes training data, game metrics, conversation history, etc.
   - Example:
     ```python
     def export(self, state, info, path):
         
     ```

## Example Usage
