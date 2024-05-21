"""This module provides a class to represent and compute connected components."""

from copy import deepcopy
from typing import Generic
import numpy as np
from graphs.undirected_graph import T, UndirectedGraph


class ConnectedComponents(Generic[T]):
    """A generic representation for connected components in an undirected graph."""

    def __init__(self, graph: UndirectedGraph[T]):
        """Compute and store the connected components of the given graph.

        :param      graph       Graph for which connected components are found
        """
        self._graph = deepcopy(graph)  # Prevent future modification of this copy

        # self.num_components (int) - Number of connected components
        # self.labels (np.ndarray) - Component number for each vertex in the graph
        self.num_components, self.labels = self.find_components(self._graph)

    def find_components(self, graph: UndirectedGraph[T]) -> tuple[int, np.ndarray]:
        """Find the connected components of the given undirected graph.

        The output array will have shape (|V|,) and store the component label for each
            vertex in the graph. These labels will range from 0 to N - 1, where N is
            the number of connected components in the graph.

        Reference: Chapter 5.6 (pg. 204) of Algorithms by Jeff Erickson (2019)

        :param      graph       Graph for which component labels are found
        :returns    (Number of components, Array of component labels for each vertex)
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
                        unexplored += graph.adjacent[u]

        # Sanity-check: 1) All vertices labeled? and 2) Last component non-empty?
        assert np.all(labels != -1), "All vertices should have a component label!"
        assert np.any(labels == component_num), "Last component should be non-empty!"

        num_components = component_num + 1  # Handles zero-indexed component numbers

        return num_components, labels

    def get_graph_copy(self) -> UndirectedGraph[T]:
        """Return a copy of the stored undirected graph."""
        return deepcopy(self._graph)

    def get_component_subgraphs(self) -> list[UndirectedGraph[T]]:
        """Export the stored connected component labels as separate subgraphs.

        :returns    List of subgraphs, one for each stored connected component
        """
        components: list[UndirectedGraph[T]] = []

        for c in range(self.num_components):
            v_indices = [i for i in range(self._graph.size_V) if self.labels[i] == c]
            vertices = [self._graph.V[v_idx] for v_idx in v_indices]

            component = UndirectedGraph[T](vertices, [])

            # Now, add the component's edges (i,j) separately
            # NOTE: Indices (v,u) are in graph.V but indices (i,j) are in component.V
            for i_idx, v_idx in enumerate(v_indices):
                for u_idx in self._graph.adjacent[v_idx]:
                    assert u_idx in v_indices, "Neighbors should share their component!"

                    j_idx = v_indices.index(u_idx)
                    component.add_edge((i_idx, j_idx))

            # Add the new component into the list of components
            components.append(component)

        return components

    def get_vertex_indices(self, component_id: int) -> set[int]:
        """Return the vertex indices in the specified connected component.

        :param      component_id    ID of the component of the returned vertices
        :returns    Set of vertex indices in the requested component
        """
        in_component = self.labels == component_id
        v_indices = {v for v in range(self._graph.size_V) if in_component[v]}

        return v_indices
