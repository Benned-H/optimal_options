"""Tests for the graph_partition module."""

import numpy as np
from graphs.undirected_graph import UndirectedGraph
from graphs.graph_partition import uniform_spanning_tree
from graphs.connectivity import is_connected


def test_uniform_spanning_tree():
    """Expect that uniform spanning trees are generated correctly."""

    min_size_V = 300  # Minimum number of vertices in a test graph
    max_size_V = 500  # Maximum number of vertices in a test graph
    number_trials = 10  # Number of graphs and spanning trees to test

    rng = np.random.default_rng()

    for _ in range(number_trials):

        # Arrange - Create a fully connected graph to be spanned

        # Randomly generate the size of the test graph
        graph_size = rng.integers(min_size_V, max_size_V, endpoint=True)

        vertex_data = list(rng.integers(0, 100, size=graph_size))

        # Create a fully-connected graph using all (i,j) pairs
        edges = [(i, j) for i in range(graph_size) for j in range(graph_size)]

        graph = UndirectedGraph[int](vertex_data, edges)

        # Act - Compute a uniform spanning tree for the example graph
        spanning_tree = uniform_spanning_tree(graph, rng)

        # Assert - Verify expected properties of the output spanning tree

        # Expect the graph and spanning tree to have the same number of vertices
        assert graph.size_V == spanning_tree.size_V, "Expected |G.V| to equal |T.V|!"

        # Expect the spanning tree T to have 2 * (|G.V| - 1) edges
        tree_size_E = 2 * (graph.size_V - 1)
        assert spanning_tree.size_E == tree_size_E, "Expected |T.E| == 2(|G.V| - 1)!"

        # Expect the spanning tree to be a tree (connected and acyclic)
        assert is_connected(spanning_tree), "Expected spanning tree to be connected!"

        # TODO - UndirectedGraph needs method: is_cyclic()
