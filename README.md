# RL

Implementing reinforcement learning from scratch — a 3-day sprint from tabular Q-learning up through deep RL.

## Day 1 — Tabular Q-learning on FrozenLake

`qlearning.py` trains an agent on [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) (Gymnasium) using a plain numpy Q-table. No neural nets, no frameworks — just the Bellman update applied step by step:

```
Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',·) − Q(s,a) ]
```

- **Epsilon-greedy** action selection with exponential decay (explore early, exploit later)
- Success rate logged every 500 episodes
- Prints the final Q-table and the learned policy as arrows on the 4×4 grid

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gymnasium numpy matplotlib
```

## Run

```bash
python qlearning.py
```

With `is_slippery=False` the agent reaches ~100% success within 2000 episodes. Flip `is_slippery=True` in the file for the stochastic version — expect a ~70% ceiling (the ice sometimes wins) and a much longer climb.
