"""This module defines functions to generate optimal behaviors for Four Rooms."""

from typing import NewType

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from optimal_behaviors.abstract_a_star import AStarPlanner
from optimal_behaviors.four_rooms_planner import FourRoomsPlanner

PathT = NewType("PathT", list[int])


def generate_tasks(env: FourRoomsEnv) -> list[tuple[int, int]]:
    """Generate all shortest-path problems (s0, g) for the given environment.

    Each "task" (i.e., shortest-path problem) includes a start state and goal state.
        Problems where s0 and g are the same state are excluded.

    :param      env     Environment used to generate a state transition graph
    :returns    List of non-trivial shortest-path problems, each a tuple (s0, g)
    """
    graph = get_transition_graph(env)

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

    path = planner.a_star(s0, set(g))

    # Sanity-check: Does the path begin with s0 and end with g?
    assert path[0] == s0, f"Path found by A* should begin with initial state {s0}!"
    assert path[-1] == g, f"Path found by A* should end with goal state {g}!"

    return path


def generate_optimal_behaviors(env: FourRoomsEnv) -> list[PathT]:
    """Generate the dataset of optimal behaviors for the given environment.

    Each "optimal behavior" is a list of states forming the optimal path for a task.
        The "states" in each output path are represented as the vertex index for the
        corresponding state within the environment's transition graph.

    :param      env     Environment for which optimal behaviors are found
    :returns    List of optimal behaviors (paths over environment graph) for all tasks
    """
    all_tasks = generate_tasks(env)  # List of (s0, g) tuples

    planner = FourRoomsPlanner()
    task_solutions = [solve_task(t, planner) for t in all_tasks]

    return task_solutions
