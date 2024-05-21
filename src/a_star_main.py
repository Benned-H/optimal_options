"""This script creates all target behaviors using A*, then visualizes them."""

import numpy as np
from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from graphs.graph_partition import uniform_spanning_tree
from optimal_behaviors.generate_behaviors import generate_optimal_behaviors


def main():
    """Run the script's main method."""
    env = FourRoomsEnv(render_mode="human", fps=60)

    rng = np.random.default_rng()

    # Compute and render the state transition graph for the Four Rooms environment
    graph = get_transition_graph(env)
    spanning_tree = uniform_spanning_tree(graph, rng)
    env.transition_graphs = [spanning_tree]
    env.reset()

    env.transition_graphs = []  # Clear the spanning tree after displaying once

    # Compute the optimal behaviors only once and keep them around
    tasks_behaviors = generate_optimal_behaviors(graph)

    while True:
        user_input = input("Press 'enter' to display another task, or 'q' to quit.\n")

        if user_input == "q":
            env.close()
            exit()

        # Otherwise, display a random task solution
        random_task_idx = rng.integers(len(tasks_behaviors))
        task, path = tasks_behaviors[random_task_idx]
        s0_idx, g_idx = task

        # Convert the task's vertex indices into (x,y) states
        s0_xy = graph.V[s0_idx]
        g_xy = graph.V[g_idx]
        path_xy = [graph.V[v_idx] for v_idx in path]

        # Prepare the environment to render the sampled task
        env.set_task(s0_xy, g_xy)
        env.path = path_xy

        env.force_render()


if __name__ == "__main__":
    main()
