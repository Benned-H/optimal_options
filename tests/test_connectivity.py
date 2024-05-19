"""Tests for the connectivity module."""

import numpy as np
from graphs.undirected_graph import UndirectedGraph
from graphs.connectivity import is_connected


def test_is_connected_fully_connected():
    """Expect that any fully connected graph is considered connected."""

    min_size_V = 100  # Minimum number of vertices
    max_size_V = 250  # Maximum number of vertices
    number_trials = 30  # Number of test graphs to create

    rng = np.random.default_rng()

    for _ in range(number_trials):

        # Arrange - Create a fully connected graph

        # Randomly generate the size of the test graph
        graph_size = rng.integers(min_size_V, max_size_V, endpoint=True)

        vertex_data = list(rng.integers(0, 100, size=graph_size))

        # Create a fully-connected graph using all (i,j) pairs
        edges = [(i, j) for i in range(graph_size) for j in range(graph_size)]

        fully_connected_graph = UndirectedGraph[int](vertex_data, edges)

        # Act/Assert - Any fully connected graph should be considered connected!
        assert is_connected(fully_connected_graph)


def test_is_connected_linear():
    """Expect that any connected chain is considered connected."""

    min_size_V = 10  # Minimum number of vertices in a chain
    max_size_V = 500  # Maximum number of vertices in a chain
    number_trials = 50  # Number of test graphs to create

    rng = np.random.default_rng()

    for _ in range(number_trials):

        # Arrange - Create a barely-connected chain

        # Randomly generate the size of the chain
        chain_size = rng.integers(min_size_V, max_size_V, endpoint=True)

        vertex_data = list(rng.integers(0, 100, size=chain_size))

        # Create a random permutation of the vertices
        chain_permutation = rng.permutation(chain_size)

        # Create edges along the chain
        edges = []
        for idx in range(len(chain_permutation) - 1):
            permuted_v = chain_permutation[idx]
            permuted_u = chain_permutation[idx + 1]
            edges.append((permuted_v, permuted_u))

        chain = UndirectedGraph[int](vertex_data, edges)

        # Act/Assert - Any connected chain should be considered connected!
        assert is_connected(chain)

        # Act 2 - Remove a single random edge from the chain
        removed_edge = chain.sample_edge(rng)
        chain.remove_edge(removed_edge)

        # Assert - Expect that removing any single edge makes the chain unconnected
        assert not is_connected(chain)
