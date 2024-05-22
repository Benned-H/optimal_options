"""This script computes the model evidence of random region-based agents."""

import numpy as np
from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from graphs.graph_partition import uniform_spanning_tree, decompose
from optimal_behaviors.generate_behaviors import generate_optimal_behaviors
from agents.region_based_agent import RegionBasedAgent
from optimal_behaviors.log_model_evidence import log_model_evidence


def main():
    """Run the script's main method."""
    env = FourRoomsEnv(render_mode="human", fps=60)

    rng = np.random.default_rng()

    # Compute and render an initial random spanning tree of the environment
    graph = get_transition_graph(env)
    spanning_tree = uniform_spanning_tree(graph, rng)
    env.transition_graphs = [spanning_tree]
    env.reset()

    # Compute the optimal behaviors only once and keep them around
    print("Computing the dataset of optimal behaviors, just a second...")
    tasks_behaviors = generate_optimal_behaviors(graph)
    optimal_behaviors = [path for (_, path) in tasks_behaviors]

    while True:
        user_input = input(
            (
                "Please input one of the following:\n"
                "\tNumber of components (int)\n"
                "\t'r' to reset the spanning tree\n"
                "\t'q' to quit\n"
            )
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
            print("Error: Unable to parse an integer!")
            continue

        # Create that number of components and render them
        components = decompose(num_components, spanning_tree, rng)
        env.transition_graphs = components.get_component_subgraphs()
        env.reset()

        # Create an HRL agent using the random graph decomposition
        agent = RegionBasedAgent(components)
        total_options = agent.num_options()

        print(f"Created an HRL agent with {total_options} total subgoal options.")

        # Compute the log model evidence of the created agent
        print("Now computing this agent's log model evidence...")
        agent_LME = log_model_evidence(agent, optimal_behaviors)

        print(f"Calculated agent's LME as: {agent_LME}")


if __name__ == "__main__":
    main()
