"""This module defines a class to evaluate the fitness of region-based agents."""

import pygad

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from optimal_behaviors.generate_behaviors import generate_optimal_behaviors
from optimal_behaviors.genetic_encoding import decode_agent
from optimal_behaviors.log_model_evidence import log_model_evidence


class FitnessEvaluator:
    """Defines the fitness function used in the genetic algorithm."""

    def __init__(self):
        """Initialize the fitness evaluator."""
        self.env = FourRoomsEnv(render_mode=None)
        self.state_space = get_transition_graph(self.env)

        tasks_paths = generate_optimal_behaviors(self.state_space)
        self.behaviors = [p for (_, p) in tasks_paths]

    def fitness(self, ga_instance: pygad.GA, solution: list[int], idx: int) -> float:
        """Evaluate the fitness of the given solution.

        :param      ga_instance     Instance of the running genetic algorithm
        :param      solution        Binary encoding of a region-based HRL agent
        :param      idx             Index of the solution in the population
        :returns    Log-Bayesian model evidence of the encoded agent on all tasks
        """
        agent = decode_agent(solution, self.state_space)

        return log_model_evidence(agent, self.behaviors)
