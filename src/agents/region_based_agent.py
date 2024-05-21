"""This module defines an HRL agent using region-based options."""

import numpy as np

from envs.four_rooms import FourRoomsEnv
from graphs.connected_components import ConnectedComponents
from options.region_option import RegionBasedOption


# TODO: Abstract class for all options-based agents!
class RegionBasedAgent:
    """An agent using options based on regions of the state space."""

    def __init__(self, env: FourRoomsEnv, regions: ConnectedComponents[np.ndarray]):
        """Initialize the agent using the given state space regions.

        :param      env         Agent's environment
        :param      regions     Connected components of the state transition graph
        """
        region_ids = [r for r in range(regions.num_components)]

        self.root_policy = None  # TODO: What form does this take!?

        # Create one region-based option per region
        self.options = [RegionBasedOption(env, regions, r_id) for r_id in region_ids]
