"""This module provides methods to partition a given undirected graph."""

from copy import deepcopy
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
) -> list[UndirectedGraph[T]]:
    """Decompose the given graph into N random connected components.

    To create N random connected components, we find a uniform spanning tree,
        then remove N - 1, and finally separate the resulting component subgraphs.

    Assertion: The number of components must be at least 1 and at most |V|

    :param      n               Number of connected components to create
    :param      graph           Undirected graph to decompose into components
    :param      rng             Random number generator (initialized elsewhere)
    :returns    List of connected components (each an UndirectedGraph[T] subgraph)
    """
    assert 1 <= n and n <= graph.size_V, "{n} is an invalid number of components!"

    spanning_tree = uniform_spanning_tree(graph, rng)

    # Remove N - 1 edges from the spanning tree
    for _ in range(n - 1):
        edge = spanning_tree.sample_edge(rng)
        spanning_tree.remove_edge(edge)

    # Separate the resulting graph into its connected components
    connected_components = separate_components(spanning_tree)

    # Sanity-check - Did we end up with N components, as expected?
    resulting_n = len(connected_components)
    assert resulting_n == n, f"Error: decompose({n}) produced {resulting_n} components!"

    return connected_components


def separate_components(graph: UndirectedGraph[T]) -> list[UndirectedGraph[T]]:
    """Separate the given undirected graph into its connected components.

    Reference: Chapter 5.6 (pg. 204) of Algorithms by Jeff Erickson (2019)

    :param      graph       Undirected graph for which connected components are found
    :returns    List of the graph's connected components, each a new subgraph
    """

    # Mark all vertices as "not yet included" in any component (indicated by -1)
    labels = np.full((graph.size_V,), -1, dtype=int)
    component_num = -1  # Increments as we reach new components

    # Reference: "CountAndLabel" algorithm from "Algorithms" (Erickson, 2019)
    for v in range(graph.size_V):
        if labels[v] == -1:  # Vertex not yet labeled; new component!
            component_num += 1

            # Run DFS on this component, labeling any unlabeled vertices
            unexplored = [v]
            while unexplored:
                u = unexplored.pop()

                if labels[u] == -1:  # Unlabeled vertex
                    labels[u] = component_num

                    for neighbor in graph.adjacent[u]:
                        unexplored.append(neighbor)

    # Sanity-check: 1) All vertices labeled? and 2) Last component non-empty?
    assert np.all(labels != -1), "All vertices should have a component label!"
    assert np.any(labels == component_num), "Last component should be non-empty!"

    # Construct each connected component from the vertex labels
    components: list[UndirectedGraph[T]] = []
    for c in range(component_num + 1):
        v_indices = [v_idx for (v_idx, _) in enumerate(graph.V) if labels[v_idx] == c]
        vertices = [graph.V[v_idx] for v_idx in v_indices]

        component = UndirectedGraph[T](vertices, [])

        # Now, add the component's edges (i,j) separately
        # NOTE: Indices (v,u) are in graph.V but indices (i,j) will be in component.V
        for i_idx, v_idx in enumerate(v_indices):
            for u_idx in graph.adjacent[v_idx]:
                assert u_idx in v_indices, "Neighbors should be in the same component!"
                j_idx = v_indices.index(u_idx)
                component.add_edge((i_idx, j_idx))

        # Add the new component into the list of components
        components.append(component)

    return components
