"""This script allows the user to play the Four Rooms domain with WASD controls."""

from gymnasium.utils.play import play
from gymnasium.spaces import Dict
from envs.four_rooms import FourRoomsEnv


def main():
    """Create the Four Rooms environment."""
    env = FourRoomsEnv(render_mode="rgb_array")

    # Define mapping from WASD to agent actions
    wasd_to_action = {
        "d": 0,
        "dw": 1,
        "w": 2,
        "wa": 3,
        "a": 4,
        "as": 5,
        "s": 6,
        "sd": 7,
    }

    # Begin manual play with Four Rooms environment using WASD controls
    play(env, transpose=False, keys_to_action=wasd_to_action, noop=8, zoom=1.0)


if __name__ == "__main__":
    main()
