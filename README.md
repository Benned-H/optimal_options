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
        - [ ] Compute optimal options using subgoal formulation
        - [ ] Create subgoals from graph decompositions
    - [ ] Portable options case - Show which "visual kernels" worked best/worst
        - [ ] Need to better define the formal structure of these "kernel" options!

- [ ] Learning curves for best/worst/no options (Fig. 1B of OBH)
    - [ ] Options case - Exactly as in the paper (x: episode, y: log solution time)
        - [ ] What does "episode" measure here?
    - [ ] Portable options case - Compare best/worst/flat (x: episode, y: log solution time)
        - [ ] Could plot every single portable option (because compactly visualizable)
    - [ ] How does this differ from Fig. 2-4 of BPO?

- [ ] Compare model evidence with search time (Fig. 1B inset of OBH)
    - [ ] Options case: Best/worst/flat/arbitrary options (x: model evidence, y: search time)
        - [ ] Compute model evidence from a set of options
    - [ ] Portable options case: Best/worst/flat options (x: model evidence, y: search time)
        - [ ] Compute model evidence from a set of portable options

- [ ] Compare learned options and portable options in Four Rooms (Fig. 2-4 in BPO)
    - [ ] Plot learning curves (x: episodes, y: actions)
    - [ ] Options case (BPO Fig. 2) - Compare perfect, learned, and no options
        - [ ] No options agents are learned using $\epsilon$-greedy SARSA($\lambda$)
        - [ ] Options are learned using off-policy trace-based tree-backup updates
            - [ ] See *[Precup et al., 2000]* for intra-option learning.
            - [ ] Agents with options are given one option per salient event
            - [ ] Parameters are given on page 897 of BPO (under Fig. 1)
        - [ ] "Perfect" options are just pre-learned options
    - [ ] Learned portable options (BPO Fig. 3) - After 0/1/2/5/10 training experiences
        - [ ] One agent-space option per salient event (e.g., three in the lightworld)
        - [ ] Value functions were learned using linear function approximation
        - [ ] Updates using gradient descent ($\alpha = 0.01$) and off-policy trace-based tree-backup
        - [ ] An option could be taken if its value function exceeded a threshold of 0.1
        - [ ] Each "training experience" was 100 episodes in a random training lightworld (other than the single evaluation lightworld)
    - [ ] Compare agents with both kinds of options over 0/1/2/5/10 training experiences (Fig. 4 of BPO)
        - [ ] Same details as above, and all problem-space value functions were discrete
