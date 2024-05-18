"""This script creates random graph decompositions of the Four Rooms state space."""

from time import sleep
from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph


def main():
    """Run the script's main method."""
    env = FourRoomsEnv(render_mode="human", fps=60)

    graph = get_transition_graph(env)
    env.transition_graphs = [graph]

    env.reset()

    while True:
        sleep(3)


if __name__ == "__main__":
    main()
