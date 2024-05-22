"""This module provides functions to convert MDPs into state transition graphs."""

import numpy as np
from envs.four_rooms import FourRoomsEnv
from graphs.undirected_graph import UndirectedGraph
from graphs.connected_components import ConnectedComponents


def get_transition_graph(env: FourRoomsEnv) -> UndirectedGraph[np.ndarray]:
    """Create a state transition graph for the given Four Rooms environment.

    The output undirected graph G = (V,E) has the form:
        - Vertices V = All valid agent (x,y) locations in the MDP's state space
        - Edges E = Bidirectional connections between all action-connected states

    TODO: Generalize over an abstract interface for MDPs!

    :param      env     MDP environment used to create transition graph
    :returns    Undirected graph representing topology of the MDP
    """

    # Find the bounds of the agent's (x,y) location in the environment
    agent_xy_space = env.observation_space["agent_xy"]

    min_x, min_y = tuple(agent_xy_space.low)  # Convert array shape (2,) into tuple
    max_x, max_y = tuple(agent_xy_space.high)

    # Create a vertex for each valid agent (x,y) location
    agent_xys: list[np.ndarray] = [
        np.array((x, y))
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
    ]
    valid_agent_xys = [xy for xy in agent_xys if env.valid_xy(xy)]

    # Initialize the vertices of the transition graph
    transition_graph = UndirectedGraph[np.ndarray](valid_agent_xys, [])

    # Create a map from (x,y) locations to indices in the vertex list
    vertex_dict = {tuple(xy): idx for (idx, xy) in enumerate(valid_agent_xys)}

    # Using the MDP's action space, find all edges in the state transition graph
    for agent_xy in valid_agent_xys:
        vertex_idx = vertex_dict[tuple(agent_xy)]

        for action_idx in range(env.action_space.n):
            new_agent_xy = env.transition(agent_xy, action_idx)

            # Only add edges between different vertices (no self-connections)
            if not np.array_equal(agent_xy, new_agent_xy):
                new_vertex_idx = vertex_dict[tuple(new_agent_xy)]

                new_edge = (vertex_idx, new_vertex_idx)
                transition_graph.add_edge(new_edge)

    return transition_graph


def entrance_states(
    regions: ConnectedComponents[np.ndarray], region_id: int
) -> set[int]:
    """Find the entrance states of the specified region of the state transition graph.

    Say S_i denotes a region of the state transition graph. Then:
        Entrances(S_i) = { s in S_i that can be transitioned into from outside S_i }

    Reference: "Hierarchical Solution of Markov Decision Processes using Macro-actions"
        by Hauskrecht et al., 1998 provided this definition to create macro-actions.

    :param      regions     Connected components of the state transition graph
    :param      region_id   ID of the region for which entrances are found
    :returns    Set of entrance states for the region (as vertex indices)
    """
    entrances: set[int] = set()

    for v_idx in regions.get_vertex_indices(region_id):
        neighbors = regions.graph.adjacent[v_idx]

        # Find the region ID for each neighbor of this vertex
        neighbor_regions = [regions.labels[n] for n in neighbors]
        entrance_regions = [r for r in neighbor_regions if r != region_id]

        # Can this vertex be entered from outside the region?
        if entrance_regions:  # If so, it corresponds to an entrance state!
            entrances.add(v_idx)

    return entrances


def exit_states(regions: ConnectedComponents[np.ndarray], region_id: int) -> set[int]:
    """Find the exit states of the specified region of the state transition graph.

    Say S_i denotes a region of the state transition graph. Then:
        Exits(S_i) = { s outside S_i that can be transitioned into from S_i }

    Reference: "Hierarchical Solution of Markov Decision Processes using Macro-actions"
        by Hauskrecht et al., 1998 provided this definition to create macro-actions.

    :param      regions     Connected components of the state transition graph
    :param      region_id   ID of the component for which exits are found
    :returns    Set of exit states for the region (as vertex indices)
    """
    exits: set[int] = set()

    in_region = regions.get_vertex_indices(region_id)
    outside_region = [v for v in range(regions.graph.size_V) if v not in in_region]

    for v_idx in outside_region:
        neighbors = regions.graph.adjacent[v_idx]

        # Find the region ID for each neighbor of this vertex
        neighbor_regions = [regions.labels[n] for n in neighbors]

        # Can this vertex be exited into from the region (region_id)?
        if region_id in neighbor_regions:  # If so, it corresponds to an exit state!
            exits.add(v_idx)

    return exits
