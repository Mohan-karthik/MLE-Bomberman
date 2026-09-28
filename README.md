# Reinforcement Learning for Bomberman — Team Blastage

Machine Learning Essentials
Summer Semester 2026


## Models

We developed three reinforcement-learning model families:

1. Tabular SARSA
2. Linear Approximate Q-Learning
3. Dueling Double DQN

The development followed a curriculum from coin navigation,
to crate destruction and bomb escape, to opponent hunting,
and finally full four-player Bomberman.

## Repository Structure

agent_code/SARSA_agent     - SARSA curriculum agents
agent_code/my_agent         - developmental Approx-Q agent
agent_code/clean_my_agent   - final learned-only Approx-Q agent
agent_code/clean_dqn_agent  - Dueling Double DQN



## Dependencies

Python 3
NumPy
PyTorch

No additional libraries beyond the provided Bomberman environment
are required for the final agents.

## Running an Agent

Example:

python main.py play --agents clean_my_agent rule_based_agent \
rule_based_agent rule_based_agent --scenario classic \
--n-rounds 100 --no-gui