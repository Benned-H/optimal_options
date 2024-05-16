"""Tests for the generic UndirectedGraph class."""

import numpy as np
from graphs.undirected_graph import UndirectedGraph
from graphs.graph_partition import uniform_spanning_tree


def test_graph_adjacency_unique():
    """Expect that the graph ignores repeated (bidirectional) edges."""

    # Vertex data doesn't need to be unique. Six vertices, indexed 0 to 5
    vertex_data = ["a", "b", "c", "d", "c", "e"]

    # Edges are bidirectional. Here, 3 unique edges * 2 directions = 6 edges total
    edges = [(0, 1), (2, 3), (1, 0), (0, 3), (2, 3)]

    graph = UndirectedGraph[str](vertex_data, edges)

    # Expect six vertices and six edges
    assert graph.size_V == 6
    assert graph.size_E == 6


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
        spanning_tree = uniform_spanning_tree(graph)

        # Assert - Verify expected properties of the output spanning tree

        # Expect the graph and spanning tree to have the same number of vertices
        assert graph.size_V == spanning_tree.size_V, "Expected |G.V| to equal |T.V|!"

        # Expect the spanning tree to have 2 * (|V| - 1) edges
        tree_size_E = 2 * (graph.size_V - 1)
        assert spanning_tree.size_E == tree_size_E, "Expected |T.E| == 2(V - 1)!"

        # Expect the spanning tree to be a tree (connected and acyclic)
        # TODO - UndirectedGraph needs methods is_connected() and is_cyclic()
