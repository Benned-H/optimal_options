"""This module implements a generic undirected graph using adjacency lists."""

import numpy as np
from typing import TypeVar, Generic

T = TypeVar("T")


class UndirectedGraph(Generic[T]):
    """A generic undirected graph composed of vertices and bidirectional edges."""

    def __init__(self, vertices: list[T], edges: list[tuple[int, int]]):
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

        # Compute and store the number of vertices and edges in the graph
        self.size_V: int = len(self.V)
        self.size_E: int = sum([len(adj) for adj in self.adjacent])

    def add_vertex(self, data: T):
        """Create a new vertex containing the given data.

        :param      data        Generic data stored in the new vertex
        """
        self.V.append(data)
        self.adjacent.append(set())  # New vertex begins with no adjacent vertices

        self.size_V += 1

    def add_edge(self, edge: tuple[int, int]):
        """Add the given edge (i,j) to the undirected graph.

        :param      edge        Edge (i,j) connecting two vertices (i and j)
        """
        (i, j) = edge

        assert 0 <= i and i < self.size_V, f"Given vertex index {i} not in graph!"
        assert 0 <= j and j < self.size_V, f"Given vertex index {j} not in graph!"

        self.adjacent[i].add(j)
        self.adjacent[j].add(i)

        self.size_E += 2

    def random_neighbor(self, u: int, rng: np.random.Generator) -> int:
        """Sample a random neighbor of the given vertex.

        :param      u           Index of the vertex to sample a neighbor for
        :param      rng         Random number generator (initialized elsewhere)
        :returns    neighbor    Index of a random neighbor of vertex u
        """
        neighbors = list(self.adjacent[u])
        random_idx = rng.integers(len(neighbors))  # Index into u's neighbors list
        neighbor = neighbors[random_idx]

        return neighbor
