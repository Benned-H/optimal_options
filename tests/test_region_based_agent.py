"""Tests for the RegionBasedAgent class."""

from envs.four_rooms import FourRoomsEnv
from envs.example_regions import create_example_regions
from agents.region_based_agent import RegionBasedAgent


def test_example_possible_actions():
    """Expect that the agent produces correct numbers of possible actions."""

    # Arrange - Create regions, an agent, and the task paths for the OBH example.
    env = FourRoomsEnv(render_mode=None)
    state_space, example_regions = create_example_regions(env)

    agent = RegionBasedAgent(state_space, example_regions)

    first_task = [38, 48, 52, 60, 70, 81]
    second_task = [49, 48, 52, 60]

    # Act - Compute the agent's possible actions over the constructed paths
    first_result_actions = agent.possible_actions(first_task)
    second_result_actions = agent.possible_actions(second_task)

    # Assert - Expect the action counts given in the OBH supplementary materials.
    first_expected = [24, 4, 2, 6, 4]
    second_expected = [15, 1, 1]

    for r1, e1 in zip(first_result_actions, first_expected):
        assert r1 == e1

    for r2, e2 in zip(second_result_actions, second_expected):
        assert r2 == e2
