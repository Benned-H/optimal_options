"""This module implements a generic undirected graph using adjacency lists."""

from typing import TypeVar, Generic
import numpy as np

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
        self.adjacent: list[set[int]] = [set() for v in self.V]
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

    def add_edge(self, edge: tuple[int, int]) -> int:
        """Add the given edge (i,j) to the undirected graph.

        :param      edge        Edge (i,j) connecting two vertices (i and j)
        :returns    Integer indicating how many edges were actually added
        """
        (i, j) = edge

        assert 0 <= i and i < self.size_V, f"Given vertex index {i} not in graph!"
        assert 0 <= j and j < self.size_V, f"Given vertex index {j} not in graph!"

        # If the edge is already in the graph, don't bother adding it!
        if j in self.adjacent[i] and i in self.adjacent[j]:
            return 0

        # Otherwise, add the new edge to the graph and return True
        self.adjacent[i].add(j)  # set.add() won't do anything on repeats
        self.adjacent[j].add(i)

        # Check that self-connections only count as one edge!
        edges_added = 1 if (i == j) else 2
        self.size_E += edges_added

        return edges_added

    def remove_edge(self, edge: tuple[int, int]):
        """Remove the given edge (and its symmetric twin) from the graph.

        :param      edge        Edge (i,j) connecting two vertices (i and j)
        """
        (i, j) = edge

        assert 0 <= i and i < self.size_V, f"Given vertex index {i} not in graph!"
        assert 0 <= j and j < self.size_V, f"Given vertex index {j} not in graph!"
        assert i in self.adjacent[j], f"Given edge {edge} not in graph!"
        assert j in self.adjacent[i], f"Given edge {edge} not in graph!"

        # Remove the edge, now that we've checked it's safe to do so
        self.adjacent[i].remove(j)
        self.adjacent[j].remove(i)

        edges_removed = 1 if (i == j) else 2
        self.size_E -= edges_removed

    def get_edge_from_idx(self, edge_idx: int) -> tuple[int, int]:
        """Find the edge (i,j) corresponding to the given integer edge index.

        For example, the "first" edge connects the lowest-index vertex that actually
            has neighbors to its lowest-index neighbor. The "last" edge would connect
            the highest-index vertex with neighbors to its highest-index neighbor.

        :param      edge_idx        Index of the edge to return
        :returns    Edge (i,j) corresponding to the given index
        """
        edges_skipped = 0  # We begin having skipped zero edges

        # Look along the vertices, and their adjacency lists, from low to high
        for current_vertex in range(self.size_V):

            neighbors = self.adjacent[current_vertex]

            if not neighbors:  # Does the current vertex have edges to consider?
                continue  # If not, skip the vertex

            # Find the edge index corresponding to this vertex's first/last neighbors
            first_neighbor_idx = edges_skipped
            last_neighbor_idx = first_neighbor_idx + len(neighbors) - 1

            # Is the requested edge index in this list of neighbors?
            here = (first_neighbor_idx <= edge_idx) and (edge_idx <= last_neighbor_idx)
            if not here:
                edges_skipped += len(neighbors)  # Skip this vertex's edges
                continue

            # Otherwise, we know that the requested edge comes from this vertex!
            neighbor_idx = 0  # Begin from first neighbor

            while edges_skipped < edge_idx:
                edges_skipped += 1  # Increment in the "global" space of edges
                neighbor_idx += 1  # Increment in the "local" space of neighbors

            # Verify a few sanity-checks before continuing...
            #   1. After that while loop, the current edge should be the one!
            #   2. The neighbor index should remain in the list of neighbors
            assert edges_skipped == edge_idx, "Expected correct edge index!"
            assert 0 <= neighbor_idx and neighbor_idx < len(neighbors)

            # Sort the neighbors list before indexing for the edge
            sorted_neighbors = sorted(list(neighbors))
            neighbor = sorted_neighbors[neighbor_idx]

            return (current_vertex, neighbor)

        # If we've somehow "missed" the desired edge, something went wrong!
        print(f"|V|: {self.size_V} and |E|: {self.size_E}")
        print(f"At the end of get_edge_from_idx({edge_idx}) but didn't find it!")
        print(f"Edges skipped: {edges_skipped}")

        assert False, "We shouldn't be here..."

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

    def sample_edge(self, rng: np.random.Generator) -> tuple[int, int]:
        """Sample a random edge from the graph.

        :param      rng         Random number generator (initialized elsewhere)
        :returns    Edge (i,j) sampled randomly from all edges in the graph
        """

        # Use |E| to sample the index of a random edge
        edge_idx = rng.integers(self.size_E)  # Index: 0 through |E| - 1

        # Find the edge corresponding to the sampled index
        edge = self.get_edge_from_idx(edge_idx)

        return edge
