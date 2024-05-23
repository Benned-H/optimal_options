"""This module provides functions to encode and decode genes for HRL agents."""

import numpy as np

from graphs.undirected_graph import UndirectedGraph, T
from graphs.connected_components import ConnectedComponents
from agents.region_based_agent import RegionBasedAgent


def encode_agent(agent: RegionBasedAgent) -> np.ndarray[int]:
    """Encode the given agent into a binary genetic encoding.

    :param      agent       Agent defined by some state space decomposition
    :returns    Array of binary integers (0 or 1) representing the agent
    """
    encoding = np.full((agent.state_space.size_E,), 0, dtype=int)

    counter = 0
    for i_idx in range(agent.state_space.size_V):
        for j_idx in agent.state_space.adjacent[i_idx]:  # All possible edges in S
            if agent.regions.share_edge(i_idx, j_idx):
                encoding[counter] = 1

            counter += 1

    # Sanity-check - Expect to have encoded one bit per edge in the state space
    assert agent.state_space.size_E == counter

    return encoding


def decode_agent(
    encoding: np.ndarray[int], state_space: UndirectedGraph[T]
) -> RegionBasedAgent:
    """Decode the given encoding into a region-based HRL agent.

    :param      encoding        Binary (0 or 1) encoding representing graph edges
    :param      state_space     State space defining the possible edges present
    :returns    Region-based agent created using the encoded connected components
    """
    connectivity_graph = UndirectedGraph[T](state_space.V, [])

    counter = 0
    for i_idx in range(state_space.size_V):
        for j_idx in state_space.adjacent[i_idx]:  # All possible neighbors of i

            if encoding[counter] == 1:
                connectivity_graph.add_edge((i_idx, j_idx))

            counter += 1

    regions = ConnectedComponents(connectivity_graph, state_space)

    return RegionBasedAgent(state_space, regions)
