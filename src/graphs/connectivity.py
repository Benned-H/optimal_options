"""This module provides functions for computing the connectivity of graphs."""

import numpy as np
from graphs.undirected_graph import T, UndirectedGraph


def is_connected(graph: UndirectedGraph[T]) -> bool:
    """Check whether the given undirected graph is connected."""
    marked = np.full((graph.size_V,), False, dtype=bool)

    # We should be able to reach all vertices from the first vertex!
    unexplored = [0]
    while unexplored:
        u = unexplored.pop()

        # Ensure that we don't infinitely recurse on cycles in the graph
        if not marked[u]:
            marked[u] = True
            unexplored += graph.adjacent[u]

    return bool(np.all(marked))
