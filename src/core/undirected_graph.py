"""This module implements a generic undirected graph using adjacency lists."""

from typing import TypeVar, Generic

T = TypeVar("T")


class UndirectedGraph(Generic[T]):
    """A generic undirected graph composed of vertices and bidirectional edges."""

    def __init__(self, vertices=list[T], edges=list[tuple[int, int]]):
        """Initialize the undirected graph, given vertex and edge lists.

        All edges (i,j) will be added as both (i,j) and (j,i) in the graph.

        :param      vertices            List of data inside each vertex
        :param      edges               List of (i,j) vertex connections
        """

        # V contains vertices 0, ..., |V - 1|; data accessed by indexing V[i]
        self.V = vertices

        # Edges are stored as adjacency sets for each vertex
        self.adjacent = [set() for v in self.V]
        for edge in edges:
            (i, j) = edge  # Each edge defines a connection between two vertices

            self.adjacent[i].add(j)
            self.adjacent[j].add(i)

    def vertex_count(self):
        """Return the number of vertices in the graph."""
        return len(self.V)

    def edge_count(self):
        """Return the number of edges in the graph."""
        return sum([len(adj) for adj in self.adjacent])
