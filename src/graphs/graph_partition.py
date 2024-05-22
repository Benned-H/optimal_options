"""This module provides methods to partition a given undirected graph."""

import numpy as np
from graphs.undirected_graph import T, UndirectedGraph
from graphs.connected_components import ConnectedComponents


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

            edges_added = tree.add_edge(new_edge)  # Add the edge to the tree
            assert edges_added == 2  # Sanity-check - Two edges should always be added!

            in_tree[u] = True
            u = next_v[u]

        # Exit early if all vertices are already in the tree
        if np.all(in_tree):
            break

    # Verify expected properties before exiting
    assert np.all(in_tree), "All nodes should be marked as in the spanning tree!"

    return tree


def decompose(
    n: int, graph: UndirectedGraph[T], rng: np.random.Generator
) -> ConnectedComponents[T]:
    """Decompose the given graph into N random connected components.

    To create N random connected components, find a uniform spanning tree, remove
        N - 1 edges, and finally compute the resulting connected components.

    Assertion: The number of components must be at least 1 and at most |V|

    :param      n               Number of connected components to create
    :param      graph           Undirected graph to decompose into components
    :param      rng             Random number generator (initialized elsewhere)
    :returns    Connected components object (contains component labels for each vertex)
    """
    assert 1 <= n, f"{n} < 1 is an invalid number of components!"
    assert (
        n <= graph.size_V
    ), f"{n} > {graph.size_V} is an invalid number of components!"

    spanning_tree = uniform_spanning_tree(graph, rng)

    # Remove N - 1 edges from the spanning tree
    for _ in range(n - 1):
        edge = spanning_tree.sample_edge(rng)
        spanning_tree.remove_edge(edge)

    # Find the connected components of the resulting graph
    connected_components = ConnectedComponents(spanning_tree)
    connected_components.graph = graph  # Store the original graph's connectivity

    # Sanity-check - Did we end up with N components, as expected?
    resulting_n = connected_components.num_components
    assert resulting_n == n, f"Error: decompose({n}) produced {resulting_n} components!"

    return connected_components
