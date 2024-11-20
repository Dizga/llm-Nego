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

3. **Step Method:**
   - Takes an action and returns a tuple `(observation, reward, done, info)`.
   - Example:
     ```python
     def step(self, action):
         # Apply action and return the new state, reward, done, and info
         return observation, reward, done, info
     ```

4. **Render Method (optional):**
   - Renders the environment for visualization.
   - Example:
     ```python
     def render(self, mode='human'):
         # Render the environment to the screen or other output
     ```

5. **Close Method (optional):**
   - Cleans up resources when the environment is no longer needed.
   - Example:
     ```python
     def close(self):
         # Clean up resources
     ```

## Example Usage
