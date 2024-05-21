"""Tests for the generate_behaviors module."""

from envs.four_rooms import FourRoomsEnv
from optimal_behaviors.four_rooms_planner import FourRoomsPlanner
from optimal_behaviors.generate_behaviors import generate_tasks, solve_task


def test_all_tasks_solvable():
    """Expect that all tasks in the Four Rooms environment produce a non-empty path."""

    # Arrange - Create Four Rooms environment, its planner, and all tasks
    env = FourRoomsEnv(render_mode=None)
    planner = FourRoomsPlanner(env)
    tasks = generate_tasks(env)

    # Act/Assert - Compute optimal path for each task, expect it to be non-empty
    for task in tasks:
        path = solve_task(task, planner)

        assert path, f"generate_optimal_behaviors() found no path for task {task}!"
