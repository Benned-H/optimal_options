"""This module defines an HRL agent implementing the options framework."""

from typing import Generic
from options.deterministic_option import StateT, ActionT, DeterministicOption


class OptionsAgent(Generic[StateT, ActionT]):
    """An agent implementing the options framework over generic state/action types."""

    def __init__(self, options: set[DeterministicOption[StateT, ActionT]]):
        """Initialize the agent using a set of generic options.

        :param      options     Set of options available to the HRL agent
        """
        self.options = options

    def available_options(self, s: StateT) -> set[DeterministicOption[StateT, ActionT]]:
        """Find which options are available at the given state.

        :param      s       Low-level state of the underlying MDP
        :returns    Set of options that can be initiated from s
        """
        pass

        return set()  # TODO: Actually implement
