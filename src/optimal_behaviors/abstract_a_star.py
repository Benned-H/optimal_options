"""This module defines an abstract A* planner over generic types."""

from typing import TypeVar, Generic
from abc import ABC, abstractmethod

StateT = TypeVar("StateT")


class Node(Generic[StateT], ABC):
    """An abstract, generic node data structure used during A* search."""

    def __init__(self, state: StateT):
        """Initialize the node using its stored state.

        :param      state       State within the space being searched
        """
        self.state = state

        # We don't initialize these upon construction in case the node is closed
        self.g = 0  # Cost-to-reach g
        self.f = 0  # Estimated cost of entire path through the node, f = g + h(s)

        self.prev = None  # Pointer to parent node during A* search

    def __str__(self) -> str:
        """Create a human-readable string representing this node."""
        return str(self.state) + f": g = {self.g}, f = {self.f}"

    @abstractmethod
    def __eq__(self, other: Node[StateT]) -> bool:
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

    def backtrack(self, node: Node[StateT]) -> list[Node[StateT]]:
        """Backtrack from the given node to find the path it represents.

        The backtracking process uses the n.prev pointers to recursively find each
            node's parent node, resulting in a path from s0 to the given node.

        :param      node            Node for which the preceding path is found
        :returns    List of nodes from the initial state to the given node
        """
        curr = node
        path = [node]  # We'll build the path in reverse order

        while curr.prev is not None:
            path.insert(0, curr.prev)
            curr = curr.prev

        return path

    def push_open_list(self, n: Node[StateT]):
        """Push the given node to the planner's open list.

        The node will only be added to the open list if it's the lowest-cost node
            to its stored state out of all nodes in the open list. That is, if there's
            an open node with the same state, the better node (based on f) is kept.

        :param      n       Node to potentially add to the planner's open list
        """
        for o_idx, o in enumerate(self.open_list):
            if o == n:  # These nodes have the same state...

                if n.f < o.f:  # New node is better!
                    self.open_list.pop(o_idx)
                    self.open_list.append(n)

                return  # The open list should only store at most one node per state

    @abstractmethod
    def neighbors(self, node: Node[StateT]) -> set[Node[StateT]]:
        """Find the neighboring nodes of the given node.

        This method simplifies the overall A* process by combining what could have been
            two functions, actions(n) and result(n, a), into one method.

        TODO: Could skip closed neighbors at the source! I get it now! Allows us to
            build the action, cost, and f/g/h into each Node from the beginning.

        TODO: This method is placed into the A* planner, not the Node class, because
            we don't want to pass the environment's dynamics to every Node. It would
            be better to have an abstract StateSpace class, but this is fine for now.

        Note: Only sets neighbor.state, but doesn't touch g, f, or prev.

        :param      node        Node during A* search whose neighbors are expanded
        :returns    Set of nodes resulting from valid actions from the given node
        """
        pass

    @abstractmethod
    def cost(self, n1: Node[StateT], n2: Node[StateT]) -> float:
        """Find the cost to move from the current node (n1) to the next node (n2).

        :param      n1          Current node an action was taken from
        :param      n2          Next node reached by that action
        :returns    Cost of the action between the two nodes
        """
        pass

    @abstractmethod
    def h(self, node: Node[StateT], G: set[StateT]) -> float:
        """Compute a heuristic estimate of the cost-to-go from the given node.

        If there are multiple states in G, the minimum-distance goal is used.

        :param      node        Node from which cost-to-go is estimated
        :param      G           Set of goal states
        :returns    Estimated cost-to-go (must be optimistic)
        """
        pass

    def a_star(self, s0: StateT, G: set[StateT]) -> list[Node[StateT]]:
        """Run A* search on the given state space search problem.

        This function works with the abstract functions outlined above.

        :param      s0      Initial state for the search problem
        :param      G       Set of goal states to be reached
        :returns    Optimal path (list of nodes) from s0 to some goal state in G
        """
        self.open_list = [Node[StateT](s0)]  # Create a node for the starting state
        self.closed_list.clear()

        while self.open_list:  # Continue until the open list is empty
            self.open_list.sort(key=lambda node: node.f)

            curr = self.open_list.pop(0)

            if curr.state in G:  # Does this node contain a goal state?
                return self.backtrack(curr)

            self.closed_list.append(curr)

            for n in self.neighbors(curr):

                # If the neighbor isn't already closed, add it to the open list
                if n not in self.closed_list:
                    n.g = curr.g + self.cost(curr, n)  # g is cost-to-reach
                    n.f = n.g + self.h(curr, G)  # f is total-cost-through
                    n.prev = curr  # Neighbor was preceded by the current node
                    self.push_open_list(n)

        # If we exit the while loop and haven't returned, search has failed
        return []  # Empty path indicates failure
