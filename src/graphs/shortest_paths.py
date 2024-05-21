"""This module provides functions to compute shortest paths for FourRoomsEnv tasks."""

import numpy as np
from envs.four_rooms import FourRoomsEnv

def target_behaviors(env: FourRoomsEnv) -> list[tuple[np.ndarray, int]]:
    """Compute target behaviors for the given environment's current goal state.

    :param      env     Environment specifying action model and goal state
    :returns    List of (state, action) tuples targeting the task goal state
    """