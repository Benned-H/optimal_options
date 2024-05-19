"""Tests for the generic UndirectedGraph class."""

import numpy as np
from graphs.undirected_graph import UndirectedGraph


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


def test_add_vertex_and_edges():
    """Expect that add_vertex() and add_edge() correctly increment |V| and |E|"""

    min_size_V = 250  # Min. vertices
    max_size_V = 500  # Max. vertices
    min_edge_samples = 1000  # Min. randomly sampled edges
    max_edge_samples = 2000  # Max. randomly sampled edges

    number_graphs = 100  # Number of random graphs used in testing

    rng = np.random.default_rng()

    for _ in range(number_graphs):

        # Arrange - Begin with an empty graph and sample its intended contents
        graph = UndirectedGraph[int]([], [])

        expected_size_V = rng.integers(min_size_V, max_size_V, endpoint=True)
        num_samples_E = rng.integers(min_edge_samples, max_edge_samples, endpoint=True)

        # Sample the chosen numbers of vertices and edges
        vertex_samples = list(rng.integers(0, 10, size=expected_size_V))

        assert (
            len(vertex_samples) == expected_size_V
        ), "Sampled wrong number of vertices!"

        edge_samples_i = list(rng.integers(expected_size_V, size=num_samples_E))
        edge_samples_j = list(rng.integers(expected_size_V, size=num_samples_E))
        edge_samples = zip(edge_samples_i, edge_samples_j)

        # Act - Add the samples into the graph
        for vertex in vertex_samples:
            graph.add_vertex(vertex)

        edges_added = 0
        for edge in edge_samples:
            edges_added += graph.add_edge(edge)  # Repeats are ignored and return 0
        expected_size_E = edges_added

        # Assert - Verify the expected sizes of the graph's V and E
        assert graph.size_V == expected_size_V, f"Expected |V| to be {expected_size_V}!"
        assert graph.size_E == expected_size_E, f"Expected |E| to be {expected_size_E}!"


def test_get_edge_from_idx():
    """Expect that the get_edge_from_idx() method correctly handles random graphs."""

    min_size_V = 100  # Min. vertices
    max_size_V = 500  # Max. vertices
    min_samples_E = 500  # Min. number of randomly sampled edges
    max_samples_E = 1000  # Max. number of randomly sampled edges

    number_graphs = 30  # Number of random graphs used in testing
    tests_per_graph = 20  # Number of random edge indices to test per graph

    rng = np.random.default_rng(2)

    for _ in range(number_graphs):

        # Arrange - Create a new random graph using the test's hyperparameters
        graph_size = rng.integers(min_size_V, max_size_V, endpoint=True)
        vertex_data = list(rng.integers(0, 10, size=graph_size))

        # Create the test graph initially without edges
        graph = UndirectedGraph[int](vertex_data, [])

        # Sample a random number of random edges
        edge_samples = rng.integers(min_samples_E, max_samples_E, endpoint=True)
        for _ in range(edge_samples):
            random_i = rng.integers(graph_size)  # Sample two random vertex indices
            random_j = rng.integers(graph_size)
            graph.add_edge((random_i, random_j))  # Repeats will be ignored

        print(f"\n\nGraph has {graph.size_V} vertices and {graph.size_E} edges.")

        # Sample a number of random edge indices and test the graph accordingly
        for _ in range(tests_per_graph):
            request_edge_idx = rng.integers(graph.size_E)

            print(f"\nRequesting edge index {request_edge_idx}...")

            # Act - Request the desired edge from the undirected graph
            result_edge = graph.get_edge_from_idx(request_edge_idx)

            print(f"Resulting edge was {result_edge}.")

            # Assert - Verify that the resulting edge correctly "lines up"
            result_i, result_j = result_edge

            edges_before_i = -1
            for before_i in range(result_i):  # Doesn't include i's edge count
                edges_before_i += len(graph.adjacent[before_i])

            # Find the expected index into vertex i's neighbors, based on the result
            expected_neighbor_idx = request_edge_idx - edges_before_i - 1

            # Find the corresponding expected neighbor (say, j) of vertex i
            sorted_neighbors_i = sorted(list(graph.adjacent[result_i]))
            expected_j = sorted_neighbors_i[expected_neighbor_idx]

            assert result_j == expected_j, "Result's neighbor didn't add up!"
