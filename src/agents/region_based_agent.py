"""This module defines an HRL agent using region-based subgoal options."""

import numpy as np

from graphs.connected_components import ConnectedComponents
from graphs.state_transition_graph import entrance_states, exit_states
from options.region_subgoal_option import RegionSubgoalOption
from optimal_behaviors.generate_behaviors import PathT


# TODO: Create interface for OBH-relevant methods later along!
class RegionBasedAgent:
    """An agent using subgoal options based on regions of the state space."""

    def __init__(self, regions: ConnectedComponents[np.ndarray]):
        """Initialize the agent using the given state space decomposition.

        :param      regions     Connected components of the state transition graph
        """
        self.regions = regions  # Store the agent-specific graph decomposition

        region_ids = [r for r in range(regions.num_components)]
        size_V = regions.graph.size_V  # To be passed to the subgoal options

        # The agent has options for each exit state (subgoal) of each region
        #   Indexing into self.options by region ID gives that region's options
        self.options: list[set[RegionSubgoalOption]] = []
        self.region_entrances: list[set[int]] = []  # Store each region's entrances
        self.region_exits: list[set[int]] = []  # Store each region's exits

        # Each region induces a set of options, one for each exit state
        for r_id in region_ids:

            # All options for the region share its entrance/exit states
            entrances = entrance_states(regions, r_id)
            self.region_entrances.append(entrances)

            exits = exit_states(regions, r_id)
            self.region_exits.append(exits)

            # Create a new region-based subgoal option for each exit of the region
            self.options.append(
                {RegionSubgoalOption(entrances, exits, e, size_V) for e in exits}
            )

        self.root_policy = None  # TODO: Initialize with real datatype!

    def num_options(self) -> int:
        """Find the total number of subgoal options this agent has available."""
        return sum([len(region_options) for region_options in self.options])

    def find_subgoals(self, path: PathT) -> np.ndarray:
        """Find the relevant subgoal for each state in the given path.

        At each state, the agent can begin (or remain within) a subgoal option
            pursuing a particular exit state somewhere in the path's future.

        :param      path        Sequence of states (each a vertex index)
        :returns    Array containing the relevant subgoal at each state (else -1)
            Note: This array will have shape (N,) where N = len(path)
        """
        assert len(path) >= 1, f"Cannot find subgoals along an empty path: {path}!"

        # Track what the agent's subgoal should be at each state in the path
        subgoals = np.full((len(path),), -1, dtype=int)  # -1 means "no subgoal"

        # From each state in the path, can the agent initiate a subgoal option?
        curr_idx = 0
        while curr_idx < len(path):

            state_v = path[curr_idx]  # Current state along the path
            curr_region = self.regions.labels[state_v]

            # Will the agent exit this region in the future?
            exits_region_in_future = False

            for future_idx in range(curr_idx + 1, len(path)):
                future_state_v = path[future_idx]
                future_region = self.regions.labels[future_state_v]

                if curr_region != future_region:  # Agent exits the region here!

                    exits_region_in_future = True

                    # This future state is the relevant exit state (subgoal) for all
                    #   states from the current up through one before future_idx
                    for towards_subgoal_idx in range(curr_idx, future_idx):
                        subgoals[towards_subgoal_idx] = future_state_v

                    # Jump to the next state without an assigned subgoal
                    curr_idx = future_idx
                    break  # Exit the loop over future_idx

            # If the agent doesn't ever exit the region, skip the entire path
            if not exits_region_in_future:
                curr_idx = len(path)

        return subgoals

    def possible_actions(self, path: PathT) -> np.ndarray:
        """Find the number of possible actions for the agent at each state in the path.

        The output array is *one shorter than the path* because the agent doesn't
            choose an action for the final state in the path.

        Each state will correspond to one of four cases:
            1. Task-specific policy - Low-level actions + available options
            2. Option-specific policy - Low-level actions
            3. Both - Possible task-policy actions * option-policy actions
            4. Neither - One possible action (intuitively, only one possible outcome)

        :param      path        Sequence of states (each a vertex index)
        :returns    Array containing the number of possible actions for each state
            Note:  This array will have shape (N - 1,) where N = len(path)
        """
        subgoals = self.find_subgoals(path)  # Relevant subgoal for each state

        # Begin each "possible actions" value as 1, representing one possible outcome.
        #   Other possible actions will be multiplied in as we proceed.
        possibilities = np.full((len(path) - 1,), 1, dtype=int)

        for idx, curr_state_v in enumerate(path[:-1]):
            curr_subgoal = subgoals[idx]
            curr_region = self.regions.labels[curr_state_v]

            # The task-specific policy is constrained on the first state (idx == 0),
            #   whenever there's no subgoal, and right after an option terminates.
            first_state = idx == 0
            no_subgoal = curr_subgoal == -1

            option_just_terminated = False
            if not first_state:  # Avoid indexing below 0
                option_just_terminated = curr_subgoal != subgoals[idx - 1]

            constrain_pi_t = first_state or no_subgoal or option_just_terminated

            # An option-specific policy is constrained whenever its subgoal is active
            #   and it hasn't been constrained for this particular state before.
            constrain_pi_o = False

            if curr_subgoal != -1:  # Some option-specific policy is active...
                region_options: set[RegionSubgoalOption] = self.options[curr_region]

                # Find this region's subgoal option for the current subgoal
                o_sg = {o for o in region_options if o.subgoal == curr_subgoal}
                assert (
                    len(o_sg) == 1
                ), f"Expected exactly one matching subgoal option, found {len(o_sg)}!"

                relevant_option = o_sg.pop()  # This option has the current subgoal

                # Has this option's policy been constrained for the current state?
                if not relevant_option.constrained[curr_state_v]:
                    constrain_pi_o = True

                    # Mark the option's policy as now constrained on this state
                    relevant_option.constrained[curr_state_v] = True

            # Now, compute the action divisor based on the constrained policies

            # Case 4 - Neither policy was constrained; continue
            if (not constrain_pi_t) and (not constrain_pi_o):
                continue  # Jump to the next state in the path

            # Otherwise, we care about the out-degree of the current state
            degree_v = len(self.regions.graph.adjacent[curr_state_v])

            if constrain_pi_t:

                # The region's options are available only if we're in an entrance
                available_options = 0
                if first_state or (curr_state_v in self.region_entrances[curr_region]):
                    available_options = len(self.options[curr_region])

                # Task-level policies can select primitive actions or available options
                task_level_available_actions = degree_v + available_options

                possibilities[idx] *= task_level_available_actions

            if constrain_pi_o:

                # Option-level policies can only select primitive actions
                option_level_available_actions = degree_v

                possibilities[idx] *= option_level_available_actions

        return possibilities
