"""This module defines an HRL agent using region-based subgoal options."""

import numpy as np

from graphs.connected_components import ConnectedComponents
from graphs.state_transition_graph import entrance_states, exit_states
from options.region_subgoal_option import RegionSubgoalOption


# TODO: Create interface for OBH-relevant methods later along!
class RegionBasedAgent:
    """An agent using subgoal options based on regions of the state space."""

    def __init__(self, regions: ConnectedComponents[np.ndarray]):
        """Initialize the agent using the given state space decomposition.

        :param      regions     Connected components of the state transition graph
        """
        region_ids = [r for r in range(regions.num_components)]

        # The agent has options for each exit state (subgoal) of each region
        #   Indexing into self.options by region ID gives that region's options
        self.options: list[set[RegionSubgoalOption]] = []

        # Each region induces a set of options, one for each exit state
        for r_id in region_ids:

            # All options for the region share its entrance/exit states
            entrances = entrance_states(regions, r_id)
            exits = exit_states(regions, r_id)

            # Create a new region-based subgoal option for each exit of the region
            self.options.append(
                {RegionSubgoalOption(entrances, exits, e) for e in exits}
            )

        self.root_policy = None  # TODO: Initialize with real datatype!
