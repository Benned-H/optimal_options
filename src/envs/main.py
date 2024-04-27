"""Creates the Four Rooms environment and conducts project's experiments."""

from time import sleep
from envs.four_rooms import FourRoomsEnv


def main():
    """Create the Four Rooms environment."""
    env = FourRoomsEnv(render_mode="human")
    print(env.reset())
    while True:
        sleep(1)


if __name__ == "__main__":
    main()
