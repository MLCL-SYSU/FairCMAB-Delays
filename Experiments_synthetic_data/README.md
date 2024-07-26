Source code using *synthetic data* for our paper "Merit-based Fair Combinatorial Semi-Bandit with Unrestricted Feedback Delays".

# Repository Structure
- data: collect data generated during the execution of the algorithms for simulation, including reward regret, fairness regret, average selection fractions, etc.
- plots: Load collected data and generate figures in the paper.
- algorithms: Detailed procedure of *FCUCB-D, FCTS_D, OP-FCUCB-D* and *OP-FCTS-D* algortihms proposed in the paper.
- arm: Define the reward distributions and feedback delay distributions of the arms.
- run: Implementation of the four algorithms.
- utilities: A collection of helper functions.

# Reproducibility
Run main.py to reproduce the experiments on synthetic data under different types of delay distributions.
