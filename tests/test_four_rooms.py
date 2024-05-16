"""Tests for the FourRoomsEnv class."""

import numpy as np
from envs.four_rooms import FourRoomsEnv


def test_reset_agent_sampling():
    """Expect that the reset() method never leaves the agent in a wall."""
    env = FourRoomsEnv(render_mode=None)

    for _ in range(5 * env.size**2):
        obs, _ = env.reset()

        agent_xy = obs["agent_xy"]  # Cartesian (x,y) coordinate
        x, y = tuple(agent_xy)

        row = env.size - 1 - y  # Convert bottom-to-top y to top-to-bottom row
        col = x  # Both x and column are left-to-right
        agent_rc = (row, col)

        assert not env.walls_rc[agent_rc]


def test_reset_goal_sampling_into_agent():
    """Expect that the reset() method never leaves the agent on the goal."""
    env = FourRoomsEnv(render_mode=None)

    for _ in range(5 * env.size**2):
        obs, _ = env.reset()

        agent_xy = tuple(obs["agent_xy"])  # Both are Cartesian (x,y) coords
        goal_xy = tuple(obs["goal_xy"])

        assert agent_xy != goal_xy


def test_reset_goal_sampling_into_walls():
    """Expect that the reset() method never leaves the goal in a wall."""
    env = FourRoomsEnv(render_mode=None)

    for _ in range(5 * env.size**2):
        obs, _ = env.reset()

        goal_xy = obs["goal_xy"]  # Cartesian (x,y) coordinate
        x, y = tuple(goal_xy)

        row = env.size - 1 - y  # Convert bottom-to-top y to top-to-bottom row
        col = x  # Both x and column are left-to-right
        goal_rc = (row, col)

        assert not env.walls_rc[goal_rc]


def test_xy_to_rc_to_pix_xy_conversion():
    """Expect that (x,y) coordinates consistently convert to (x,y) pixels."""
    env = FourRoomsEnv(render_mode=None)
    all_xy = [np.array([x, y]) for x in range(env.size) for y in range(env.size)]

    # Check correct conversion for every (x,y) coordinate in the environment
    for coord_xy in all_xy:
        direct_pix_xy = env.xy_to_pix_xy(coord_xy)  # Convert to pixel (x,y)

        index_rc = env.xy_to_rc(coord_xy)  # Convert to (r,c) index...
        through_rc_pix_xy = env.rc_to_pix_xy(index_rc)  # then to pixel (x,y)

        assert tuple(direct_pix_xy) == tuple(through_rc_pix_xy)
