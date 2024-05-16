# Optimal Portable Options

A Bayesian model selection framework for identifying optimal portable options.

## Installation

To create a conda environment to run this repository, use the commands:
```bash
conda env create -f environment.yml
conda activate opo-env
```

## Testing

To run the repository's tests, use the command:
```bash
pytest --durations=3
```

## Work Remaining

To complete this project, the following things will be needed:

- [ ] Visualize the "best" options and portable options (Fig. 1A of OBH)
    - [ ] Options case - Show which nodes were subgoals
        - [ ] Represent decompositions of the undirected graph
        - [ ] Create subgoals from graph decompositions
        - [ ] Define an `Option` class
        - [ ] Compute model evidence from a set of options
        - [ ] Compute optimal options using subgoal formulation
    - [ ] Portable options case - Show which "visual kernels" worked best/worst
        - [ ] Compute model evidence from a set of portable options
        - [ ] Need to better define the formal structure of these "kernel" options!

- [ ] Compare model evidence with search time (Fig. 1B inset of OBH)
    - [ ] Options case: Best/worst/flat/arbitrary options (x: model evidence, y: search time)
        - [ ] Compute model evidence from a set of options
    - [ ] Portable options case: Best/worst/flat options (x: model evidence, y: search time)
        - [ ] Compute model evidence from a set of portable options

- [ ] Learning curves for options, portable options, and both (Fig. 1B in OBH, Fig. 2-4 in BPO)
    - These are all in the Four Rooms environment, initially
    - [ ] Plot learning curves: (x: episodes, y: (log) solution time)
        - "Episodes" - Each episode trains the agent from the start state to the goal
        - "Solution time" - Number of steps/actions to reach the goal
    - [ ] Options case - Compare perfect, learned, and no options ("flat")
        - See OBH Fig. 1B and BPO Fig. 2
        - [ ] Learn options over best/worst/arbitrary structures
            - [ ] How does OBH "arbitrary" compare with BPO "learned" options?
        - [ ] "Flat" agents are learned using $\epsilon$-greedy SARSA($\lambda$) (BPO) or actor-critic (Botvinick et al., 2008)
        - [ ] Options are learned using off-policy trace-based tree-backup updates (BPO) or actor-critic
            - [ ] See *[Precup et al., 2000]* for intra-option learning
                - [ ] Parameters are given on page 897 of BPO (under Fig. 1)
            - [ ] *[Botvinick et al., 2008]* used actor-critic to learn options
        - [ ] "Perfect" options are just pre-learned options
    - [ ] Portable options case - Compare best and worst sets of portable options
        - [ ] See BPO Fig. 3 - After 0/1/2/5/10 training experiences on *other* worlds
            - [ ] Each "training experience" was 100 episodes in a random training world (different from the single evaluation world)
        - [ ] Experiment with 1, 2, or 4 portable options. Start with 2x2 kernels
        - [ ] Could plot every single portable option (because compactly visualizable)
        - [ ] In BPO, value functions were learned using linear function approximation
        - [ ] Updates using gradient descent ($\alpha = 0.01$) and off-policy trace-based tree-backup
            - An option could be taken if its value function exceeded a threshold of 0.1
    - [ ] Both types case - Compare agents with both kinds of options over 0/1/2/5/10 training experiences
        - See Fig. 4 of BPO
        - [ ] Same details and learning methods as above
    - Across this experiment, the "structure" of the options, portable or not, is fixed to a particular set of subgoals (nodes or kernels). We can evaluate the efficiency of learning under these different sets of options by comparing performance (solution time) versus the number of training episodes a method has received.
