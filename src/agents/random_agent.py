"""This module implements an agent that selects random actions."""

import gymnasium as gym


class RandomAgent:
    """An agent that selects random actions from the environment's action space."""

    def __init__(self, env: gym.Env):
        """Initialize the random agent for a particular environment.

        :param      env     Agent's environment
        """
        self.env = env

    def get_action(self) -> int:
        """Return a random action from the environment's action space."""
        return self.env.action_space.sample()
