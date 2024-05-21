"""Tests for the generate_behaviors module."""

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from optimal_behaviors.four_rooms_planner import FourRoomsPlanner
from optimal_behaviors.generate_behaviors import (
    generate_tasks,
    solve_task,
    generate_optimal_behaviors,
)


def test_all_tasks_solvable():
    """Expect that all tasks in the Four Rooms environment produce a non-empty path."""

    # Arrange - Create Four Rooms environment, transition graph, planner, and tasks
    env = FourRoomsEnv(render_mode=None)
    graph = get_transition_graph(env)

    planner = FourRoomsPlanner(graph)
    tasks = generate_tasks(graph)

    # Act/Assert - Compute optimal path for each task, expect it to be non-empty
    for task in tasks:
        path = solve_task(task, planner)

        assert path, f"Expected to find a non-empty path for task {task}!"


def test_generate_optimal_behaviors():
    """Expected that all generated optimal behavior paths are non-empty."""

    # Arrange - Create Four Rooms environment and its transition graph
    env = FourRoomsEnv(render_mode=None)
    graph = get_transition_graph(env)

    # Act - Compute all optimal behaviors for the state transition graph
    tasks_paths = generate_optimal_behaviors(graph)

    # Assert - Expect that all paths are non-empty
    for task, path in tasks_paths:
        assert path, f"Expected to find a non-empty path for task {task}!"
