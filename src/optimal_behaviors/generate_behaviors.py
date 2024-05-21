"""This module defines functions to generate optimal behaviors for Four Rooms."""

from typing import NewType
import numpy as np

from graphs.undirected_graph import UndirectedGraph
from optimal_behaviors.abstract_a_star import AStarPlanner
from optimal_behaviors.four_rooms_planner import FourRoomsPlanner

PathT = NewType("PathT", list[int])
TaskT = NewType("TaskT", tuple[int, int])


def generate_tasks(graph: UndirectedGraph[np.ndarray]) -> list[TaskT]:
    """Generate all shortest-path problems (s0, g) for the given graph.

    Each "task" (i.e., shortest-path problem) includes a start state and goal state.
        Problems where s0 and g are the same state are excluded.

    :param      graph       State transition graph for the underlying MDP
    :returns    List of non-trivial shortest-path problems, each a tuple (s0, g)
    """

    # Generate all (s0, g) pairs, then filter out trivial problems (s0 == g)
    all_pairs = [(s0, g) for s0 in range(graph.size_V) for g in range(graph.size_V)]
    nontrivial_tasks = [(s0, g) for (s0, g) in all_pairs if s0 != g]

    return nontrivial_tasks


def solve_task(s0_g: tuple[int, int], planner: AStarPlanner[int]) -> PathT:
    """Find the optimal behavior for the given (s0, g) task.

    :param      s0_g        Tuple containing (initial state, goal state)
    :param      planner     Path planner based on A* search
    :returns    Path (list of vertex indices) from s0 to g
    """
    s0, g = s0_g

    path = planner.a_star(s0, {g})

    if path:  # Sanity-check: Does the path begin with s0 and end with g?
        assert path[0] == s0, f"Path found by A* should begin with initial state {s0}!"
        assert path[-1] == g, f"Path found by A* should end with goal state {g}!"
    else:
        print(f"No path found for task (s0 = {s0}, g = {g})")

    return path


def generate_optimal_behaviors(
    graph: UndirectedGraph[np.ndarray],
) -> list[tuple[TaskT, PathT]]:
    """Generate the dataset of optimal behaviors for the given graph.

    Each "optimal behavior" is a list of states forming the optimal path for a task.
        These "states" correspond to vertex indices in the state transition graph.

    :param      graph       State transition graph for the underlying MDP
    :returns    List of (task, optimal behavior path) tuples
    """
    all_tasks = generate_tasks(graph)  # List of (s0, g) tuples

    planner = FourRoomsPlanner(graph)
    task_solutions = [solve_task(t, planner) for t in all_tasks]

    tasks_paths = [(all_tasks[i], task_solutions[i]) for i in range(len(all_tasks))]

    return tasks_paths
