"""This script visualizes the vertex number of each state in the Four Rooms domain."""

import numpy as np
from pygame.font import SysFont

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph


def main():
    """Show the vertex number of each state in the environment."""
    env = FourRoomsEnv(render_mode="human")

    graph = get_transition_graph(env)
    env.transition_graphs = [graph]
    env.reset()  # Initializes pygame (including pygame.font)

    env.show_vertex_idx = True
    env.font = SysFont("ubuntumono", size=28)

    env.reset()

    rng = np.random.default_rng()

    # Loop: Select random start/goal states, then render and print them
    while True:
        new_s0 = rng.integers(graph.size_V)
        new_g = rng.integers(graph.size_V)

        env.set_task(graph.V[new_s0], graph.V[new_g])
        env.force_render()

        print(f"New s0 index: {new_s0}, New goal index: {new_g}")

        user_input = input("Press 'enter' to sample new vertices, 'q' to quit.\n")

        if user_input == "q":
            env.close()
            exit()


if __name__ == "__main__":
    main()
