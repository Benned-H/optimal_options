"""Tests for the genetic_encoding module."""

import numpy as np

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from graphs.graph_partition import decompose
from agents.region_based_agent import RegionBasedAgent
from optimal_behaviors.genetic_encoding import encode_agent, decode_agent


def test_encode_decode():
    """Expect that region-based agents are the same after encoding and decoding."""

    min_num_regions = 1  # Minimum number of regions to create
    max_num_regions = 20  # Maximum number of regions to create
    num_agents = 5  # Number of random agents to sample per region count

    # Arrange - Create the random number generator
    env = FourRoomsEnv(render_mode=None)
    graph = get_transition_graph(env)

    rng = np.random.default_rng()

    # Sample N agents for each number of regions
    for num_regions in range(min_num_regions, max_num_regions + 1):
        for _ in range(num_agents):

            # Act - Sample a random agent, encode it, and decode that encoding
            components = decompose(num_regions, graph, rng)
            agent = RegionBasedAgent(graph, components)

            encoding = encode_agent(agent)
            decoded_agent = decode_agent(encoding, graph)

            # Assert - Expect that the two agents are equal after decoding
            assert agent == decoded_agent, "Agents should be equal after decoding!"
