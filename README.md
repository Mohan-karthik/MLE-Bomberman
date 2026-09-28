# Reinforcement Learning for Bomberman — Team Blastage

**Machine Learning Essentials**  
**Summer Semester 2026**

---

## Overview

In this project, we developed and evaluated three reinforcement-learning approaches for the Bomberman environment.

Our development followed a curriculum from simple navigation to full four-player gameplay:

**Coin Navigation → Crate Destruction → Bomb Escape → Opponent Hunting → Full Multiplayer**

---

## Models

We developed three reinforcement-learning model families:

1. **Tabular SARSA**
   - Used as our initial interpretable reinforcement-learning approach.
   - Developed progressively through navigation, bombing, opponent hunting, and multiplayer stages.

2. **Linear Approximate Q-Learning**
   - Extended the tabular approach using engineered features and learned linear Q-values.
   - Progressed from the developmental `my_agent` to the final learned-only `clean_my_agent`.

3. **Dueling Double DQN**
   - Used a neural-network Q-function with engineered state features.
   - Included experience replay, a target network, and Double DQN training.

---

## Repository Structure

```text
agent_code/
│
├── SARSA_agent/
│   └── SARSA curriculum agents and trained models
│
├── my_agent/
│   └── Developmental Approximate Q-Learning agent
│
├── clean_my_agent/
│   └── Final learned-only Approximate Q-Learning agent
│
└── clean_dqn_agent/
    └── Dueling Double DQN agent
```

---

## Agent Development

### SARSA

The SARSA agent was developed through a curriculum that gradually introduced more difficult parts of the Bomberman environment:

```text
Task 1 → Coin navigation
Task 2 → Crates, bombs, and escape
Task 3 → Opponent hunting
Task 4 → Full multiplayer gameplay
```

### Approximate Q-Learning

The Approximate Q-Learning agent was developed through several feature stages:

```text
22 features
    ↓
36 features
    ↓
50 features
    ↓
64 features
    ↓
Clean learned-only policy
```

The original `my_agent` combined learned Q-values with additional hand-written tactical logic during development.

The final `clean_my_agent` removes those inference-time tactical bonuses and selects actions using:

```text
Game State
    ↓
Engineered Features
    ↓
Learned Q-Values
    ↓
Mechanical Legality Mask
    ↓
argmax(Q)
```

### Dueling Double DQN

The DQN agent uses a **75-feature state representation** and a Dueling Double DQN architecture.

```text
75 Input Features
        ↓
    128 ReLU
        ↓
    128 ReLU
      ↙     ↘
 Value     Advantage
 Head        Head
  ↓           ↓
  1        6 Actions
      ↘     ↙
       Q-Values
```

---

## Dependencies

The project uses:

- **Python 3**
- **NumPy**
- **PyTorch**

The agents are designed to run inside the provided Bomberman framework.

If additional dependencies are required, they are listed in `requirements.txt`.

---

## Running an Agent

From the root directory of the Bomberman project:

```bash
python main.py play \
  --agents clean_my_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --n-rounds 100 \
  --no-gui
```

For Windows PowerShell, the command can also be written on one line:

```powershell
python main.py play --agents clean_my_agent rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 100 --no-gui
```

---

## Training an Agent

Example training command:

```powershell
python main.py play --agents clean_my_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds 1000 --no-gui
```

Training behavior and hyperparameters differ between the SARSA, Approximate Q-Learning, and DQN agents.

---

## Evaluation

The main evaluation metrics used during development included:

- Total score
- Coins collected
- Opponents killed
- Suicides / deaths
- Crates destroyed
- Survival rate
- Wins and ties
- Invalid actions
- Runtime per action

Agents were tested against progressively stronger opponents, including:

```text
peaceful_agent
coin_collector_agent
rule_based_agent
```

---
