"""This module implements a subgoal option based on a region of the state graph."""

from typing import NewType
import numpy as np

from options.deterministic_option import DeterministicOption

StateV = NewType("StateV", int)  # Represents a vertex index in the transition graph


class RegionSubgoalOption(DeterministicOption[StateV]):
    """A subgoal option based on a region of the state transition graph."""

    def __init__(
        self,
        entrances: set[StateV],
        exits: set[StateV],
        subgoal: StateV,
        num_states: int,
    ):
        """Initialize a subgoal option given its entrances, exits, and subgoal.

        :param      entrances       Entrance states of the region (as vertex indices)
        :param      exits           Exit states of the region (as vertex indices)
        :param      subgoal         Vertex index of the option's subgoal
        :param      num_states      Size of the state space for this option
        """
        self.entrances = entrances  # Defines the option's initiation set
        self.exits = exits  # Defines the option's termination set
        self.subgoal = subgoal

        # Used to track which states have been constrained for the option's policy
        self.constrained = np.full((num_states,), False, dtype=bool)

    def can_initiate(self, s: StateV) -> bool:
        """Check if the option can be initiated at the given state.

        A region-based subgoal option can initiate at its region's entrance states.

        :param      s       Low-level state of the underlying MDP
        :returns    Boolean indicating whether s is in the option's initiation set.
        """
        return s in self.entrances

    def pi(self, s: StateV) -> int:
        """Get the option policy's action for the given state.

        The policy for a region-based subgoal option is only defined over its region.

        TODO: Representation TBD

        :param      s       Low-level state of the underlying MDP
        :returns    Action index selected by the option's policy at state s.
        """
        return -1  # TODO: Design, implement, and document!

    def terminates_at(self, s: StateV) -> bool:
        """Check whether the option should terminate at the given state.

        A region-based subgoal option should terminate at its region's exit states.

        Note: Unlike some formulations of options [1], this function is deterministic.

        :param      s       Low-level state of the underlying MDP
        :returns    Boolean indicating whether the option should terminate at s.
        """
        return s in self.exits
