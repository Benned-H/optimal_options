"""This script reads the results JSON files for the optimal region-based options."""

import fnmatch
import os
import json
import numpy as np

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from graphs.connected_components import ConnectedComponents
from optimal_behaviors.genetic_encoding import decode_agent


def main():
    """Render the best-found region-based subgoal options."""
    env = FourRoomsEnv(render_mode="human")
    env.skip_agent_goal = True

    graph = get_transition_graph(env)

    for filename in os.listdir("results"):
        if fnmatch.fnmatch(filename, "*-69710*"):
            filepath = os.path.join("results", filename)

            f = open(filepath)
            data = json.load(f)

            # Extract the encoded region-based agent
            encoding = np.array(data["solution"])

            agent = decode_agent(encoding, graph)

            env.transition_graphs = agent.regions.get_component_subgraphs()
            env.reset()

            input(f"Showing results for file: {filename}. Press 'enter' to continue.")

            f.close()


if __name__ == "__main__":
    main()
