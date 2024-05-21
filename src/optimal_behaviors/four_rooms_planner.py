"""This module defines a concrete A* planner for the Four Rooms environment."""

from typing import NewType
import numpy as np

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from optimal_behaviors.abstract_a_star import Node, AStarPlanner

# We'll perform A* search over vertex indices in the transition graph
StateV = NewType("StateV", int)


class FourRoomsNode(Node[StateV]):
    """A concrete node to structure A* search in the Four Rooms environment."""

    def __init__(self, state: StateV, prev: Node[StateV], a_cost: float, h: float):
        """Initialize the node using its stored state and A*-relevant data.

        :param      state       State to be stored in this node
        :param      prev        Previous node during A* search
        :param      a_cost      Cost of the action from the previous node to this node
        :param      h           Heuristic estimate of cost-to-go from this node
        """
        super().__init__(state, prev, a_cost, h)

    def __str__(self) -> str:
        """Create a human-readable string representing this node."""
        return f"State: {int(self.state)}, g: {self.g}, f: {self.f}"

    def __eq__(self, other: Node[StateV]) -> bool:
        """Check whether the given node contains the same state.

        Overrides the `==` operator for these objects, allowing "node equality" to
            ignore the internal cost-to-reach g and estimated total cost f.

        :param      other       Another node to compare equality against
        :returns    Boolean indicating if the nodes' stored states are equal
        """
        return self.state == other.state


class FourRoomsPlanner(AStarPlanner[StateV]):
    """A concrete A* planner for the Four Rooms environment."""

    def __init__(self, env: FourRoomsEnv):
        """Initialize the A* planner using the abstract class constructor.

        :param      env     Four Rooms environment defining state transition graph
        """
        self.transition_graph = get_transition_graph(env)

        super().__init__()

    def unclosed_neighbors(self, node: Node[StateV]) -> list[Node[StateV]]:
        """Find the unclosed neighboring nodes of the given node.

        For the Four Rooms environment, neighboring states during A* will correspond
            to neighboring vertex indices in the state transition graph.

        TODO: This method is placed into the A* planner, not the Node class, because
            we don't want to pass the environment's dynamics to every Node. It would
            be better to have an abstract StateSpace class, but this is fine for now.

        :param      node        Node during A* search whose neighbors are expanded
        :returns    List of unclosed nodes resulting from valid actions from the node
        """
        neighbors = self.transition_graph.adjacent[node.state]
        unclosed_neighbors = [n_v for n_v in neighbors if not self.state_closed(n_v)]

        # Now convert the neighboring states (v_idxs) into Node objects
        neighbor_nodes = []
        for neighbor_state in unclosed_neighbors:
            action_cost = self.cost(node.state, neighbor_state)
            h_value = self.h(neighbor_state, self.goals)
            neighbor_node = FourRoomsNode(neighbor_state, node, action_cost, h_value)
            neighbor_nodes.append(neighbor_node)

        return neighbor_nodes

    def cost(self, s1: StateV, s2: StateV) -> float:
        """Find the cost to move from one state (s1) to another state (s2).

        In the Four Rooms environment, assume action cost is Euclidean distance.

        :param      s1          Current state an action was taken from
        :param      s2          Next state reached by that action
        :returns    Cost of the action between the two states
        """
        s1_xy = self.transition_graph.V[s1]  # These states represent vertex indices
        s2_xy = self.transition_graph.V[s2]

        distance_m = np.linalg.norm(s1_xy - s2_xy)

        return distance_m

    def h(self, state: StateV, goals: set[StateV]) -> float:
        """Compute a heuristic estimate of the cost-to-go from the given state.

        If there are multiple goal states, the minimum-distance goal is used. In the
            Four Rooms environment, Euclidean distance is used as the heuristic.

        TODO: Separate class for heuristic function, naturally.

        :param      state       State from which cost-to-go is estimated
        :param      goals       Set of goal states
        :returns    Estimated cost-to-go (must be optimistic)
        """
        assert len(goals) >= 1, "Estimating h(s) requires at least one goal!"

        state_xy = self.transition_graph.V[state]  # Convert v_idx into (x,y)
        goals_xy = [self.transition_graph.V[v_idx] for v_idx in goals]

        distances_m = [np.linalg.norm(state_xy - g_xy) for g_xy in goals_xy]
        min_distance_m = min(distances_m)

        return min_distance_m

    def reset(self, s0: StateV, goals: set[StateV]):
        """Set up the A* planner for search on the given planning problem.

        The open list begins as [s0] and the closed list begins empty.

        :param      s0      Initial state in the planning problem
        :param      goals   Set of goal states to be reached
        """
        self.open_list = [FourRoomsNode(s0, None, 0.0, self.h(s0, goals))]
        self.closed_list.clear()
        self.goals = goals
