"""This module defines functions to compute the log model evidence of HRL agents."""

import numpy as np

from agents.region_based_agent import RegionBasedAgent
from optimal_behaviors.generate_behaviors import PathT


def log_model_evidence(agent: RegionBasedAgent, behaviors: list[PathT]) -> float:
    """Compute the log model evidence (LME) of the given HRL agent.

    :param      agent           HRL agent representing a particular set of options
    :param      behaviors       Dataset of optimal target behaviors (i.e., paths)
    :returns    LME of the agent's options given the dataset of target behaviors
    """
    running_sum = 0.0  # Sum over all paths in the dataset

    for path in behaviors:
        possible_actions = agent.possible_actions(path)
        running_sum -= np.sum(np.log(possible_actions))  # Element-wise log, then sum

    return running_sum
