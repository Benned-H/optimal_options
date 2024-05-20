"""This script creates random graph decompositions of the Four Rooms state space."""

import numpy as np
from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from graphs.graph_partition import uniform_spanning_tree, decompose


def main():
    """Run the script's main method."""
    env = FourRoomsEnv(render_mode="human", fps=60)

    # Compute the state transition graph for the Four Rooms environment
    graph = get_transition_graph(env)

    # Render the initial transition graph before computing any decompositions
    env.transition_graphs = [graph]

    env.reset()

    input("Press 'enter' to create and render the initial spanning tree.\n")

    rng = np.random.default_rng()

    spanning_tree = uniform_spanning_tree(graph, rng)
    env.transition_graphs = [spanning_tree]
    env.reset()

    while True:
        user_input = input(
            "Please input one of the following:\n\tNumber of components (int)\n\t'r' to reset the spanning tree\n\t'q' to quit\n"
        )

        if user_input == "q":
            env.close()
            exit()
        elif user_input == "r":  # Reset the spanning tree!
            spanning_tree = uniform_spanning_tree(graph, rng)
            env.transition_graphs = [spanning_tree]
            env.reset()
            continue

        try:  # Otherwise, try to parse an integer
            num_components = int(user_input)
        except ValueError:
            print("Error: Please input an integer, or 'r'!")
            continue

        components = decompose(num_components, spanning_tree, rng)
        env.transition_graphs = components.get_component_subgraphs()
        env.reset()


if __name__ == "__main__":
    main()
