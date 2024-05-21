"""This module implements an option based on a region of the state transition graph."""

from typing import NewType
import numpy as np

from envs.four_rooms import FourRoomsEnv
from options.deterministic_option import DeterministicOption
from graphs.connected_components import ConnectedComponents
from graphs.state_transition_graph import entrance_states, exit_states

StateXY = NewType("StateXY", np.ndarray)
Action = NewType("Action", int)


class RegionBasedOption(DeterministicOption[StateXY, Action]):
    """A subgoal option based on a region of the state transition graph."""

    def __init__(
        self,
        env: FourRoomsEnv,
        regions: ConnectedComponents[np.ndarray],
        region_id: int,
    ):
        """Initialize an option based on the specified region of the state space.

        TODO: Generalize over multiple environments (not just FourRoomsEnv)

        :param      env         Four Rooms environment defining the action space
        :param      regions     Connected components of the state transition graph
        :param      region_id   Index of the region upon which the option is based
        """
        self.env = env  # Store the environment

        # Compute and store the region's entrance and exit states
        self.region_entrances: set[StateXY] = entrance_states(regions, region_id)
        self.region_exits: set[StateXY] = exit_states(regions, region_id)

    def can_initiate(self, s: StateXY) -> bool:
        """Check if the option can be initiated at the given state.

        A region-based option can initiate in any of the region's entrance states.

        TODO: Should this also consider the task's start state? OBH says yes...

        :param      s       Low-level state of the underlying MDP
        :returns    Boolean indicating whether s is in the option's initiation set.
        """
        return s in self.region_entrances

    def pi(self, s: StateXY) -> Action:
        """Get the option policy's action for the given state.

        TODO: How should this actually be implemented? See the papers!

        :param      s       Low-level state of the underlying MDP
        :returns    Low-level action selected by the option's policy at state s.
        """
        action = self.env.action_space.sample()  # TODO: Don't actually act randomly!

        return action

    def terminates_at(self, s: StateXY) -> bool:
        """Check whether the option should terminate at the given state.

        A region-based option terminates at all exit states of the region.

        Note: Unlike some formulations of options [1], this function is deterministic.

        :param      s       Low-level state of the underlying MDP
        :returns    Boolean indicating whether the option should terminate at s.
        """
        return s in self.region_exits
