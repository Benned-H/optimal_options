"""This module implements the Four Rooms gridworld using Gymnasium.

The original rooms problem was described by Sutton et al. (1999) [1], although
    we use the adapted version presented by Botvinick et al. (2009) [2].

References:

[1] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A
    framework for temporal abstraction in reinforcement learning,”
    Artificial Intelligence, vol. 112, no. 1, pp. 181–211, Aug. 1999,
    doi: 10.1016/S0004-3702(99)00052-1.

[2] M. M. Botvinick, Y. Niv, and A. G. Barto, “Hierarchically organized
    behavior and its neural foundations: A reinforcement learning perspective,”
    Cognition, vol. 113, no. 3, pp. 262–280, Dec. 2009,
    doi: 10.1016/j.cognition.2008.08.011.
"""

import numpy as np
import pygame

import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete


class FourRoomsEnv(gym.Env):
    """A deterministic Four Rooms RL domain."""

    # Support human-friendly render mode
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 13):
        """Initialize the Four Rooms environment.

        :param      render_mode     Selected render mode of the environment
        :param      size            Size of the square grid
        """
        self.size = size  # Size of the square grid
        self.window_size = 512  # Size of the PyGame window (pixels)

        """
        Observations include the agent's and goal's (x,y) locations, which
            range from 1 to size - 2 (due to environment outer walls).
        """
        self.observation_space = Dict(
            {
                "agent": Box(low=1, high=size - 2, shape=(2,), dtype=int),
                "goal": Box(low=1, high=size - 2, shape=(2,), dtype=int),
            }
        )

        # Eight possible actions move the agent into an adjacent square
        self.action_space = Discrete(8)

        # Maps abstract action indices to the corresponding (x,y) direction
        self._action_to_direction = {
            0: np.array([1, 0]),  # Right
            1: np.array([1, 1]),  # Up-Right
            2: np.array([0, 1]),  # Up
            3: np.array([-1, 1]),  # Up-Left
            4: np.array([-1, 0]),  # Left
            5: np.array([-1, -1]),  # Down-Left
            6: np.array([0, -1]),  # Down
            7: np.array([1, -1]),  # Down-Right
        }

        # Ensure that render_mode is None, or supported by the environment
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be the window drawn to,
            and `self.clock` will be a clock used to render at the correct FPS.

        Both will remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Translate the environment's state into an observation."""
        return {"agent": self._agent_xy, "goal": self._goal_xy}

    def _get_info(self):
        """Provide auxiliary information for the step() and reset() methods."""
        # For now, there's not really anything to return.
        return {}

    def reset(self, seed=None):
        """Reset the environment to an initial state for a new episode.

        :param      seed        Random seed to initialize environment's RNG
        """
        # Seed the RNG for parent gymnasium.Env
        super().reset(seed=seed)

        # Choose agent's starting (x,y) location uniformly at random
        # TODO: Ensure agent and goal don't collide with walls!
        self._agent_xy = self.observation_space["agent"].sample()

        # Sample goal's (x,y) location until it doesn't collide with the agent
        self._goal_xy = self._agent_xy
        while np.array_equal(self._goal_xy, self._agent_xy):
            self._goal_xy = self.observation_space["goal"].sample()

        comment = """TODO: Reference for creating the walls

        # Begin with an empty grid surrounded by four walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Create center vertical walls (x = 6; y = 0 to 2, 4 to 9, 11 to 12)
        self.grid.vert_wall(6, 0, 3)
        self.grid.vert_wall(6, 4, 6)
        self.grid.vert_wall(6, 11, 2)

        # Create left horizontal walls (x = 0 to 1, 3 to 6; y = 6)
        self.grid.horz_wall(0, 6, 2)
        self.grid.horz_wall(3, 6, 4)

        # Create right horizontal walls (x = 6 to 8, 10 to 12; y = 7)
        self.grid.horz_wall(6, 7, 3)
        self.grid.horz_wall(10, 7, 3)
        """
        print(comment)  # TODO: Delete dummy variable!

        # Create initial observation and information, then render if needed
        initial_obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return initial_obs, info

    def step(self, action_index: int):
        """Compute the new state of the environment after the given action.

        :param      action_index    Index of the action to be simulated
        """
        # Map the action index to the correspoding movement
        movement = self._action_to_direction[action_index]

        # Compute what the next state would be, if valid
        new_agent_xy = self._agent_xy + movement

        # Check if the new state is valid (inside grid and no wall collisions)
        # TODO: Create walls and add collision checking for them!
        agent_in_bounds = (1 <= new_agent_xy) & (new_agent_xy <= self.size - 2)

        # Update the state only if the agent's new location is valid
        if agent_in_bounds.all():
            self._agent_xy = new_agent_xy

        # Episode ends iff the agent reaches the goal
        terminated = np.array_equal(self._agent_xy, self._goal_xy)
        reward = 100 if terminated else 0  # Binary sparse reward

        # Compute new observation and information, then render if needed
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        """Compute the render frames specified by `self.render_mode`."""
        if self.render_mode is None:
            return None

        # Continual rendering should be called during step(), not here
        if self.render_mode == "human":
            return None

    def _render_frame(self):
        """Render a frame representing the current state of the environment."""

        # Initialize the window if it hasn't been initialized
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Default to white background
        cell_pixels = self.window_size / self.size  # Size of grid cell (pixels)
        cell_rect = (cell_pixels, cell_pixels)

        # Draw the goal in green
        pygame.draw.rect(
            canvas, (18, 181, 32), pygame.Rect(self._goal_xy * cell_pixels, cell_rect)
        )

        # Draw the agent in red: 176, 23, 59
        pygame.draw.circle(
            canvas,
            (176, 23, 59),
            (self._agent_xy + 0.5) * cell_pixels,  # Center of the circle
            cell_pixels / 2.5,  # Radius (pixels)
        )

        # Add gridlines throughout the environment (in black)
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, cell_pixels * x),  # Start position (x,y)
                (self.window_size, cell_pixels * x),  # End position (x,y)
                width=3,
            )

            pygame.draw.line(
                canvas,
                0,
                (cell_pixels * x, 0),  # Start position (x,y)
                (cell_pixels * x, self.window_size),  # End position (x,y)
                width=3,
            )

        # In "human" mode, copy the drawing from `canvas` to the visible window
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Keep the human-rendering at a stable framerate (delays if early)
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Close all open resources used by the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
