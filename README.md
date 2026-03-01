# Introduction to Reinforcement Learning Demos

This repository contains three Python demos aligned with your RL guide:

- **Policy Optimization (REINFORCE)**
- **Q-learning (tabular)**
- **Gymnasium environment solving with PPO**

The diagram `rl_algorithms_9_15.svg` can be used as a visual companion while running the scripts.

## Prerequisites

- Python 3.10 or 3.11
- `uv` installed
- Existing virtual environment at `.venv` (already created in your workspace)

## Install dependencies

From the project root:

```powershell
uv pip install -r requirements.txt
```

## Demos

### 1) REINFORCE on CartPole (Gymnasium)

Script: `demos/reinforce_cartpole.py`

```powershell
python demos/reinforce_cartpole.py
```

Useful options:

```powershell
python demos/reinforce_cartpole.py --episodes 1500 --lr 0.0005 --gamma 0.99
```

Visualize learned policy after training:

```powershell
python demos/reinforce_cartpole.py --render-eval --render-episodes 3
```

Record evaluation episodes to video (for slides):

```powershell
python demos/reinforce_cartpole.py --record-video --video-episodes 3 --video-dir videos/reinforce_cartpole
```

Do both in one command:

```powershell
python demos/reinforce_cartpole.py --record-and-render --render-episodes 3 --video-episodes 3
```

What it shows:

- Monte Carlo policy gradient update
- Return normalization for stable learning
- Running average of episode reward

### 2) Tabular Q-learning on FrozenLake

Script: `demos/q_learning_frozenlake.py`

```powershell
python demos/q_learning_frozenlake.py
```

Deterministic map (default) trains faster for a classroom demo.

Stochastic version (harder, more realistic):

```powershell
python demos/q_learning_frozenlake.py --slippery
```

Visualize learned policy after training:

```powershell
python demos/q_learning_frozenlake.py --render-eval --render-episodes 5
```

Record evaluation episodes to video (for slides):

```powershell
python demos/q_learning_frozenlake.py --record-video --video-episodes 5 --video-dir videos/q_learning_frozenlake
```

Do both in one command:

```powershell
python demos/q_learning_frozenlake.py --record-and-render --render-episodes 5 --video-episodes 5
```

What it shows:

- Epsilon-greedy exploration
- Bellman update for Q-table
- Final policy quality via success-rate evaluation

### 3) Solve Gymnasium CartPole with PPO

Script: `demos/gymnasium_ppo_cartpole.py`

```powershell
python demos/gymnasium_ppo_cartpole.py
```

This trains a PPO agent using `stable-baselines3`, saves it to `models/ppo_cartpole`, and prints evaluation metrics.

Optional local render after training:

```powershell
python demos/gymnasium_ppo_cartpole.py --render-demo
```

Render evaluation episodes (recommended):

```powershell
python demos/gymnasium_ppo_cartpole.py --render-eval --render-episodes 3
```

Record evaluation episodes to video (for slides):

```powershell
python demos/gymnasium_ppo_cartpole.py --record-video --video-episodes 3 --video-dir videos/ppo_cartpole
```

Do both in one command:

```powershell
python demos/gymnasium_ppo_cartpole.py --record-and-render --render-episodes 3 --video-episodes 3
```

## Suggested teaching flow

1. Start with `q_learning_frozenlake.py` (value-based, tabular).
2. Move to `reinforce_cartpole.py` (direct policy optimization).
3. Finish with `gymnasium_ppo_cartpole.py` (modern practical algorithm and library usage).

## Additional documentation

- Detailed explanation (EN): `docs/rl_explanations.md`
- Detailed explanation (ES): `docs/rl_explanations.es.md`

## Quick RL Guide coverage

The repository docs also map to the core theory in `RL Guide.pdf`:

- RL fundamentals: agent, environment, state, action, reward, policy.
- Model-free vs model-based RL (why this repo focuses on model-free demos).
- Practical contrast between policy optimization and Q-learning.
- Clear distinction between model, policy, and value function.

## Notes

- `requirements.txt` now explicitly includes `gymnasium` for the demo scripts.
- Legacy `gym` is still present to preserve compatibility with older code in your modernization path.
