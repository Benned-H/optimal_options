"""Creates the Four Rooms environment and conducts project's experiments."""

import optimal_portable_options


def main():
    """Create the Four Rooms environment."""
    env = gym.make("BotvinickFourRooms-v0", render_mode="human")
    print(env.reset())


if __name__ == "__main__":
    main()
