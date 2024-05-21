"""This module defines an abstract A* planner over generic types."""

from typing import TypeVar, Generic
from abc import ABC, abstractmethod

StateT = TypeVar("StateT")


class Node(Generic[StateT], ABC):
    """An abstract, generic node used to structure A* search."""

    def __init__(self, state: StateT, prev: "Node[StateT]", a_cost: float, h: float):
        """Initialize the node using its stored state and A*-relevant data.

        :param      state       State to be stored in this node
        :param      prev        Previous node during A* search
        :param      a_cost      Cost of the action from the previous node to this node
        :param      h           Heuristic estimate of cost-to-go from this node
        """
        self.state = state

        self.prev = prev  # Pointer to parent node during A* search

        prev_g = 0  # Handle special case for initial state (when prev is None)
        if self.prev is not None:
            prev_g = prev.g

        self.g = prev_g + a_cost  # Cost to reach this node
        self.f = self.g + h  # Estimated total cost through this node

    @abstractmethod
    def __repr__(self) -> str:
        """Create an unambiguous string representation for this node."""
        pass

    @abstractmethod
    def __eq__(self, other: "Node[StateT]") -> bool:
        """Check whether the given node contains the same state.

        Overrides the `==` operator for these objects, allowing "node equality" to
            ignore the internal cost-to-reach g and estimated total cost f.

        :param      other       Another node to compare equality against
        :returns    Boolean indicating if the nodes' stored states are equal
        """
        pass


class AStarPlanner(Generic[StateT], ABC):
    """An abstract A* planner over a generic state representation."""

    def __init__(self):
        """Initialize the A* planner's necessary member variables."""
        self.open_list: list[Node[StateT]] = []
        self.closed_list: list[Node[StateT]] = []
        self.goals: set[StateT] = set()

    def backtrack(self, node: Node[StateT]) -> list[StateT]:
        """Backtrack from the given node to find the path it represents.

        The backtracking process uses the n.prev pointers to recursively find each
            node's parent node, resulting in a path from s0 to the given node's state.

        :param      node            Node for which the preceding path is found
        :returns    List of states from the initial state to the given node's state
        """
        curr_node = node
        path = [node.state]  # We'll build the path in reverse order

        while curr_node.prev is not None:
            curr_node = curr_node.prev
            path.insert(0, curr_node.state)

        return path

    def push_open_list(self, n: Node[StateT]):
        """Push the given node to the planner's open list.

        The node will be added if it contains a new state, or if it's the lowest-cost
            node containing its state out of all nodes in the open list (based on f).

        :param      n       Node to potentially add to the planner's open list
        """

        for o_idx, o in enumerate(self.open_list):
            if o == n:  # These nodes have the same state...

                if n.f < o.f:  # Node is better; remove the old one!
                    self.open_list.pop(o_idx)
                    break
                else:  # Otherwise, this node was worse and we can exit
                    return

        # If here, either there was no node with the same state, or this node won!
        self.open_list.append(n)

    def state_closed(self, state: StateT):
        """Check whether any node in the closed list contains the given state.

        :param      state       State that may or may not already be closed
        :returns    Boolean indicating if the state has been closed during A* search
        """
        return any([node for node in self.closed_list if node.state == state])

    @abstractmethod
    def unclosed_neighbors(self, node: Node[StateT]) -> list[Node[StateT]]:
        """Find the unclosed neighboring nodes of the given node.

        This method simplifies the overall A* process by combining what could have been
            two functions, actions(n) and result(n, a), into one method.

        TODO: This method is placed into the A* planner, not the Node class, because
            we don't want to pass the environment's dynamics to every Node. It would
            be better to have an abstract StateSpace class, but this is fine for now.

        :param      node        Node during A* search whose neighbors are expanded
        :returns    List of unclosed nodes resulting from valid actions from the node
        """
        pass

    @abstractmethod
    def cost(self, s1: StateT, s2: StateT) -> float:
        """Find the cost to move from one state (s1) to another state (s2).

        :param      s1          Current state an action was taken from
        :param      s2          Next state reached by that action
        :returns    Cost of the action between the two states
        """
        pass

    @abstractmethod
    def h(self, state: StateT, goals: set[StateT]) -> float:
        """Compute a heuristic estimate of the cost-to-go from the given state.

        If there are multiple goal states, the minimum-distance goal is used.

        TODO: Separate class for heuristic function, naturally.

        :param      state       State from which cost-to-go is estimated
        :param      goals       Set of goal states
        :returns    Estimated cost-to-go (must be optimistic)
        """
        pass

    @abstractmethod
    def reset(self, s0: StateT, goals: set[StateT]):
        """Set up the A* planner for search on the given planning problem.

        The open list begins as [s0] and the closed list begins empty.

        :param      s0      Initial state in the planning problem
        :param      goals   Set of goal states to be reached
        """
        pass

    def a_star(self, s0: StateT, goals: set[StateT]) -> list[StateT]:
        """Run A* search on the given state space search problem.

        The output solution is a path (list of states) from s0 to some goal state,
            or an empty list when no path could be found.

        This function works with the abstract functions outlined above.

        :param      s0      Initial state for the search problem
        :param      goals   Set of goal states to be reached
        :returns    Optimal path (list of states) from s0 to some goal state, or []
        """
        self.reset(s0, goals)  # Create a node for the starting state and store goals

        while self.open_list:  # Continue until the open list is empty
            self.open_list.sort(key=lambda node: node.f)

            curr = self.open_list.pop(0)

            if curr.state in goals:  # Does this node contain a goal state?
                return self.backtrack(curr)

            self.closed_list.append(curr)

            # For each neighbor that isn't already closed, add it to the open list
            for n in self.unclosed_neighbors(curr):
                self.push_open_list(n)

        # If we exit the while loop and haven't returned, search has failed
        return []  # Empty path indicates failure
