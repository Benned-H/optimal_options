"""This script runs a genetic algorithm over agents using the PyGAD library."""

from datetime import datetime
import json
import os

import numpy as np
from pygad import GA

from envs.four_rooms import FourRoomsEnv
from graphs.state_transition_graph import get_transition_graph
from graphs.graph_partition import decompose
from agents.region_based_agent import RegionBasedAgent
from optimal_behaviors.genetic_encoding import encode_agent
from optimal_behaviors.fitness_evaluator import FitnessEvaluator


def generation_callback(ga_instance: GA):
    """Print something each generation of the genetic algorithm."""
    print(f"Generations completed: {ga_instance.generations_completed}\n")


def main():
    """Create and run a genetic algorithm over region-based HRL agents."""

    # Hyperparameters
    num_options = 4
    gens = 1  # TODO: Should halt after 20 with no change!
    gen_size = 100  # Paper used 2000

    env = FourRoomsEnv(render_mode=None)
    graph = get_transition_graph(env)

    print("Environment and state transition graph now created.")

    # Create an initial population of the above size
    population: list[list[int]] = []
    rng = np.random.default_rng()

    print(f"\nCreating population of {gen_size} agents...")

    for _ in range(gen_size):
        decomposition = decompose(num_options, graph, rng)

        # Create an HRL agent using the random graph decomposition
        agent = RegionBasedAgent(graph, decomposition)

        # Encode the agent into the initial population
        encoding = encode_agent(agent)
        population.append(encoding)

    print(f"Population now created, contains {len(population)} agents.")

    # Create the object used to evaluate agent fitness
    fitness_eval = FitnessEvaluator()

    ga_instance = GA(
        num_generations=gens,
        num_parents_mating=2,
        fitness_func=fitness_eval.fitness,
        initial_population=population,
        random_mutation_min_val=0,  # Enforce binary genes (int: 0 or 1)
        random_mutation_max_val=2,
        mutation_by_replacement=True,
        gene_type=int,
        mutation_type="random",  # Genes will mutate randomly; could also try "swap"
        on_generation=generation_callback,
    )

    print("Now running the genetic algorithm...")
    start_time = datetime.now()

    ga_instance.run()

    end_time = datetime.now()
    print(f"Genetic algorithm finished after {end_time - start_time}!")

    # Look at what the genetic algorithm produced!
    solution, best_fitness, _ = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {best_fitness}")

    results_dict = {
        "solution": solution.tolist(),
        "LME": best_fitness,
        "generations": gens,
        "population size": gen_size,
    }

    # Write the results to JSON
    filename = (
        f"results_{end_time}_LME:{int(best_fitness)}_{gens}gens_{gen_size}pop.json"
    ).replace(" ", "_")
    filepath = os.path.join("results", filename)

    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, indent=4)

    ga_instance.plot_fitness()


if __name__ == "__main__":
    main()
