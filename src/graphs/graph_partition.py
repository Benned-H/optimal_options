"""This module provides methods to partition a given undirected graph."""

import numpy as np
from graphs.undirected_graph import T, UndirectedGraph


def uniform_spanning_tree(
    graph: UndirectedGraph[T], rng: np.random.Generator
) -> UndirectedGraph[T]:
    """Compute a uniform spanning tree using Wilson's algorithm.

    A spanning tree T of graph G is a tree subgraph that includes all vertices in G.

    A uniform spanning tree of graph G is a spanning tree chosen uniformly
        at random from all possible spanning trees of G.

    :param      graph   Undirected graph for which a uniform spanning tree is found
    :param      rng     Random number generator (initialized elsewhere)
    :returns    Undirected graph representing a random spanning tree of G
    """

    # Begin with an empty tree (as if none of the vertices are in the tree)
    tree = UndirectedGraph[T](graph.V, [])
    in_tree = np.full((graph.size_V), False, dtype=bool)  # Boolean array of shape (V,)

    # Sample a random root for the tree and mark it as in the tree
    root = rng.integers(graph.size_V)
    in_tree[root] = True

    # Track the next vertex during random walks back to the tree
    next_v = np.full((graph.size_V), -1, dtype=int)  # -1 means "not initialized"

    # Generate a random permutation of the vertex indices in G
    for i in rng.permutation(graph.size_V):
        u = i  # First, random walk from i until we reach a vertex in the tree

        while not in_tree[u]:
            next_v[u] = graph.random_neighbor(u, rng)  # Overwrite cycles if they occur
            u = next_v[u]

        u = i  # Then, retrace the walk and add it to the tree
        while not in_tree[u]:
            new_edge = (u, next_v[u])
            tree.add_edge(new_edge)  # Add the edge to the output tree object

            in_tree[u] = True
            u = next_v[u]

        # Exit early if all vertices are already in the tree
        if np.all(in_tree):
            break

    # Verify expected properties before exiting
    assert np.all(in_tree), "All nodes should be marked as in the spanning tree!"

    return tree
