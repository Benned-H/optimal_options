"""This script creates a random agent in the Four Rooms environment."""

from envs.four_rooms import FourRoomsEnv
from agents.random_agent import RandomAgent


def main():
    """Run the script's main method."""
    env = FourRoomsEnv(render_mode="human", fps=100)

    random_agent = RandomAgent(env)

    env.reset()

    while True:
        action = random_agent.get_action()
        obs, reward, terminated, _, info = env.step(action)

        if terminated:
            obs, info = env.reset()


if __name__ == "__main__":
    main()
