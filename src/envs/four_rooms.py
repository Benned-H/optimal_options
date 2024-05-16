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

    # Support human-friendly and RGB array render modes
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, fps=None):
        """Initialize the Four Rooms environment.

        :param      render_mode     Selected render mode of the environment
        :param      fps             Frames per second to render the environment
        """
        self.size = 13  # Size of the square grid (always 13)
        self.window_size = 512  # Size of the PyGame window (pixels)

        # Declare agent and goal (x,y) locations in Cartesian space
        self._agent_xy = None
        self._goal_xy = None

        # Create and populate array representing walls in the environment
        walls_array = np.full((self.size, self.size), False)  # [rows, cols]

        # Surround the environment with walls
        walls_array[0, :] = True
        walls_array[-1, :] = True
        walls_array[:, 0] = True
        walls_array[:, -1] = True

        # Create center vertical walls (row = 0 to 2, 4 to 9, 11 to 12; col = 6)
        walls_array[:3, 6] = True
        walls_array[4:10, 6] = True
        walls_array[11:, 6] = True

        # Create left horizontal walls (row = 6; col = 0 to 1, 3 to 6)
        walls_array[6, :2] = True
        walls_array[6, 3:7] = True

        # Create right horizontal walls (row = 7; col = 6 to 8, 10 to 12)
        walls_array[7, 6:9] = True
        walls_array[7, 10:] = True

        # Indexed by [row, column] using standard matrix conventions
        # Rows are ordered top-to-bottom, columns ordered left-to-right
        self.walls_rc = walls_array

        """
        Observations include the agent's and goal's (x,y) locations in Cartesian
            space, with values ranging from 1 to 11 (due to the outer walls).
        """
        self.observation_space = Dict(
            {
                "agent_xy": Box(low=1, high=self.size - 2, shape=(2,), dtype=int),
                "goal_xy": Box(low=1, high=self.size - 2, shape=(2,), dtype=int),
            }
        )

        # Nine possible actions: move the agent into an adjacent square or wait
        self.action_space = Discrete(9)

        # Maps abstract action indices to (x,y) directions in Cartesian space
        self._action_to_direction_xy = {
            0: np.array([1, 0]),  # Right
            1: np.array([1, 1]),  # Up-Right
            2: np.array([0, 1]),  # Up
            3: np.array([-1, 1]),  # Up-Left
            4: np.array([-1, 0]),  # Left
            5: np.array([-1, -1]),  # Down-Left
            6: np.array([0, -1]),  # Down
            7: np.array([1, -1]),  # Down-Right
            8: np.array([0, 0]),  # No-op
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

        # Override the `fps` metadata using argument, if provided
        if fps is not None:
            self.metadata["render_fps"] = fps

    def _get_obs(self):
        """Translate the environment's state into an observation."""
        return {"agent_xy": self._agent_xy, "goal_xy": self._goal_xy}

    def _get_info(self):
        """Provide auxiliary information for the step() and reset() methods.

        See _get_obs() for information summarizing the environment's state.
        """
        return {}  # For now, there's not really anything to return.

    def xy_to_rc(self, location_xy: np.ndarray) -> np.ndarray:
        """Convert a Cartesian (x,y) coordinate into (row, col) indices.

        Notation: (x,y) coordinates increase L-to-R, bottom-to-top
                  (r,c) coordinates increase top-to-bottom, L-to-R

            The leftmost x-coordinate (0) and column (0) are equal.
            But the lowest y-coordinate (0) and row (12) aren't quite
                offset by exactly the environment's size. Instead:

            row = size - 1 - y (e.g., consider the last row, 12 as y = 0)

        :param      location_xy     Cartesian (x,y) coordinate of shape (2,)
        :returns    index_rc        Index in (row, col) space of shape (2,)
        """
        assert location_xy.shape == (2,)
        return np.array([self.size - 1 - location_xy[1], location_xy[0]])

    def rc_to_pix_xy(self, index_rc: np.ndarray) -> np.ndarray:
        """Convert a (row, col) index into an (x,y) pixel coordinate.

        Notation: (r,c) coordinates increase top-to-bottom, L-to-R
                  (x,y) pixel coordinates increase L-to-R, top-to-bottom

        :param      index_rc            Index in (row, col) space of shape (2,)
        :returns    location_pix_xy     Pixel (x,y) coordinate of shape (2,)
        """
        assert index_rc.shape == (2,)
        return np.flip(index_rc)

    def xy_to_pix_xy(self, location_xy: np.ndarray) -> np.ndarray:
        """Convert a Cartesian (x,y) coordinate into an (x,y) pixel coordinate.

        Notation: (x,y) coordinates increase L-to-R, bottom-to-top
                  (x,y) pixel coordinates increase L-to-R, top-to-bottom

        :param      location_xy         Cartesian (x,y) coordinate of shape (2,)
        :returns    location_pix_xy     Pixel (x,y) coordinate of shape (2,)
        """
        assert location_xy.shape == (2,)
        return np.array([location_xy[0], self.size - 1 - location_xy[1]])

    def wall_collision(self, location_xy: np.ndarray) -> bool:
        """Check whether the given (x,y) location collides with the room walls.

        :param      location_xy     Cartesian (x,y) coordinate of shape (2,)
        :returns    Boolean indicating if the location collides with a wall
        """
        assert location_xy.shape == (2,)
        location_rc = self.xy_to_rc(location_xy)  # Convert to (row, col) space
        return self.walls_rc[tuple(location_rc)]

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state for a new episode.

        Expect that the agent's and goal's locations are re-sampled into
            non-wall locations, and not colliding with each other.

        :param      seed        Random seed to initialize environment's RNG
        :param      options     Required for method override (unused)
        :returns    Tuple containing (initial observation, debug info)
        """
        # Seed the RNG for parent gymnasium.Env
        super().reset(seed=seed)

        # Sample agent's initial (x,y) location uniformly, avoiding walls
        self._agent_xy = self.observation_space["agent_xy"].sample()
        while self.wall_collision(self._agent_xy):
            print(f"Agent sampled into walls at {self._agent_xy}!")
            self._agent_xy = self.observation_space["agent_xy"].sample()

        # Sample goal's (x,y) location until collision-free with agent and walls
        self._goal_xy = self.observation_space["goal_xy"].sample()
        while np.array_equal(self._goal_xy, self._agent_xy) or self.wall_collision(
            self._goal_xy
        ):
            print(f"Goal sampled into collision at {self._goal_xy}!")
            self._goal_xy = self.observation_space["goal_xy"].sample()

        # Create initial observation and information, then render if needed
        initial_obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return initial_obs, info

    def step(self, action: int):
        """Compute the new state of the environment after the given action.

        :param      action      Index of the action to be simulated
        :returns    Tuple containing (obs, reward, terminated, truncated, info)
        """
        # Map the action index to the corresponding movement
        movement_xy = self._action_to_direction_xy[action]

        # Compute what the next state would be, if valid
        new_agent_xy = self._agent_xy + movement_xy

        # Check if the new state is valid (inside grid and no wall collisions)
        agent_in_bounds = (1 <= new_agent_xy) & (new_agent_xy <= self.size - 2)
        no_wall_collisions = not self.wall_collision(new_agent_xy)

        # Update the state only if the agent's new location is valid
        if agent_in_bounds.all() and no_wall_collisions:
            self._agent_xy = new_agent_xy

        # Episode ends iff the agent reaches the goal
        terminated = np.array_equal(self._agent_xy, self._goal_xy)
        reward = 100 if terminated else 0  # Binary sparse reward

        # Compute new observation and information, then render if needed
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        # Add the most recent action to the info dictionary
        info["last_action"] = action

        return observation, reward, terminated, False, info

    def render(self):
        """Compute the render frames specified by `self.render_mode`."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

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

        # Draw the walls in grey
        for wall_rc in np.argwhere(self.walls_rc):
            wall_pix_xy = self.rc_to_pix_xy(wall_rc)
            pygame.draw.rect(
                canvas,
                (92, 89, 82),
                pygame.Rect(wall_pix_xy * cell_pixels, cell_rect),
            )

        # Draw the goal in green
        goal_pix_xy = self.xy_to_pix_xy(self._goal_xy)
        pygame.draw.rect(
            canvas, (18, 181, 32), pygame.Rect(goal_pix_xy * cell_pixels, cell_rect)
        )

        # Draw the agent in red
        agent_pix_xy = self.xy_to_pix_xy(self._agent_xy)
        pygame.draw.circle(
            canvas,
            (176, 23, 59),
            (agent_pix_xy + 0.5) * cell_pixels,  # Center of the circle
            cell_pixels / 2.5,  # Radius (pixels)
        )

        # Add gridlines throughout the environment (in black)
        for line in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, cell_pixels * line),  # Start position (x,y)
                (self.window_size, cell_pixels * line),  # End position (x,y)
                width=3,
            )

            pygame.draw.line(
                canvas,
                0,
                (cell_pixels * line, 0),  # Start position (x,y)
                (cell_pixels * line, self.window_size),  # End position (x,y)
                width=3,
            )

        # In "human" mode, copy the drawing from `canvas` to the visible window
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Keep the human-rendering at a stable framerate (delays if early)
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.pixels3d(canvas))

    def close(self):
        """Close all open resources used by the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
