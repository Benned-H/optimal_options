"""This module defines an abstract, generic class representing a deterministic option.

Reference: Between MDPs and semi-MDPs (Sutton et al., 1999) [1].

[1] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A framework
    for temporal abstraction in reinforcement learning,” Artificial Intelligence,
    vol. 112, no. 1, pp. 181–211, Aug. 1999, doi: 10.1016/S0004-3702(99)00052-1.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

StateT = TypeVar("StateT")


class DeterministicOption(Generic[StateT], ABC):
    """An abstract option (extended HRL action) over generic states."""

    @abstractmethod
    def can_initiate(self, s: StateT) -> bool:
        """Check if the option can be initiated at the given state.

        :param      s       Low-level state of the underlying MDP
        :returns    Boolean indicating whether s is in the option's initiation set.
        """
        pass

    @abstractmethod
    def pi(self, s: StateT) -> int:
        """Get the option policy's action for the given state.

        :param      s       Low-level state of the underlying MDP
        :returns    Action index selected by the option's policy at state s.
        """
        pass

    @abstractmethod
    def terminates_at(self, s: StateT) -> bool:
        """Check whether the option should terminate at the given state.

        Note: Unlike some formulations of options [1], this function is deterministic.

        :param      s       Low-level state of the underlying MDP
        :returns    Boolean indicating whether the option should terminate at s.
        """
        pass
