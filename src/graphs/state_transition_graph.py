"""This module provides functions to turn MDPs into state transition graphs."""

import numpy as np
from envs.four_rooms import FourRoomsEnv
from graphs.undirected_graph import UndirectedGraph


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
