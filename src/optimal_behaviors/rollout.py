"""This module defines a rollout of optimal target behavior."""

from typing import NewType
import numpy as np

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph

StateXY = NewType("StateXY", np.ndarray)

class OptimalPath():
    """An optimal path from a particular task's start state to its goal state."""

    def __init__(self, env: FourRoomsEnv, s0: int, g: int):
        """Find the optimal path for the given task and environment.

        TODO: Make generic across environment, state, and action types!

        :param      env     Four rooms environment defining transition dynamics
        :param      s0      Index of the vertex of the start state
        :param      g       Index of the vertex of the goal state
        """
        self.transition_graph = get_transition_graph(env)
    
    def solve_task(self, s0: int, g: int):
        """Find the optimal path from s0 to g in the stored undirected graph.

        :param      s0      Start state of the path
        :param      a       
        
        """