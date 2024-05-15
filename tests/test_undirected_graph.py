"""Tests for the generic UndirectedGraph class."""

from core.undirected_graph import UndirectedGraph


def test_graph_adjacency_unique():
    """Expect that the graph ignores repeated (bidirectional) edges."""

    # Vertex data doesn't need to be unique. Six vertices, indexed 0 to 5
    vertex_data = ["a", "b", "c", "d", "c", "e"]

    # Edges are bidirectional. Here, 3 unique edges * 2 directions = 6 edges total
    edges = [(0, 1), (2, 3), (1, 0), (0, 3), (2, 3)]

    graph = UndirectedGraph[str](vertex_data, edges)

    # Expect six vertices and six edges
    assert graph.vertex_count() == 6
    assert graph.edge_count() == 6
