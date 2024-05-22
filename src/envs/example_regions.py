"""This module provides a function to create an example graph decomposition."""

import numpy as np

from envs.four_rooms import FourRoomsEnv
from graphs.connected_components import ConnectedComponents
from graphs.state_transition_graph import get_transition_graph
from graphs.graph_partition import decompose


def create_example_regions(env: FourRoomsEnv) -> ConnectedComponents[np.ndarray]:
    """Create and return the example from the supplementary material of OBH."""

    rng = np.random.default_rng()

    # Create the connected components object to be overwritten manually
    graph = get_transition_graph(env)
    components = decompose(4, graph, rng)
    components.labels.fill(-1)  # Clear all region labels

    # Manually create the intended regions
    for v_idx, v_xy in enumerate(components.graph.V):
        x, y = tuple(v_xy)

        if x < 6 and y <= 6:  # Region 0
            components.labels[v_idx] = 0
        elif x >= 6 and y <= 5:  # Region 1
            components.labels[v_idx] = 1
        elif x <= 6 and y > 6:  # Region 2
            components.labels[v_idx] = 2
        elif x > 6 and y > 5:  # Region 3
            components.labels[v_idx] = 3

        # Sanity-check - All vertices should have a component label by now
        assert (
            components.labels[v_idx] != -1
        ), f"Didn't expect label for vertex {v_idx} at xy({x},{y}) to be -1"

    return components
