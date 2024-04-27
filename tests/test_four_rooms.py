"""Tests for the FourRoomsEnv class."""

import numpy as np
from envs.four_rooms import FourRoomsEnv


def test_reset_agent_sampling():
    """Expect that the reset() method never leaves the agent in a wall."""
    env = FourRoomsEnv(render_mode=None)

    for i in range(10 * 13**2):
        obs, _ = env.reset()

        agent_xy = obs["agent_xy"]
        agent_rc = np.flip(agent_xy)  # Flips (x,y) into (r,c)

        assert not env.walls_rc[tuple(agent_rc)]


def test_reset_goal_sampling_into_agent():
    """Expect that the reset() method never leaves the agent on the goal."""
    env = FourRoomsEnv(render_mode=None)

    for i in range(10 * 13**2):
        obs, _ = env.reset()

        agent_xy = tuple(obs["agent_xy"])
        goal_xy = tuple(obs["goal_xy"])

        assert agent_xy != goal_xy


def test_reset_goal_sampling_into_walls():
    """Expect that the reset() method never leaves the goal in a wall."""
    env = FourRoomsEnv(render_mode=None)

    for i in range(10 * 13**2):
        obs, _ = env.reset()

        goal_xy = obs["goal_xy"]
        goal_rc = np.flip(goal_xy)  # Flips (x,y) into (r,c)

        assert not env.walls_rc[tuple(goal_rc)]
