# SARSA vs Q-Learning on MountainCarContinuous-v0

Reinforcement Learning — **Temporal-Difference Control**

This project implements two tabular TD control algorithms — **SARSA** (on-policy) and **Q-Learning** (off-policy) — from scratch and compares them on Gymnasium's `MountainCarContinuous-v0`. Since the environment has continuous states and actions, an adapter class discretises both so that a plain Q-table can be used.


## What's in the notebook

| Section | Content |
|---|---|
| 1. OOP Implementation | `MountainCarContinuousAdapter`, `SARSAAgent`, `QLearningAgent`, `Visualiser` |
| 2. Algorithm Details | Update rules for SARSA and Q-Learning |
| 3. Experiments | SARSA vs Q-Learning, effect of α, effect of ε, effect of discretisation, comparison with previous assignments, Q-value heatmap |
| 4. Design & Results Analysis | Summary of design choices and results |
| 5. Conceptual Questions | TD vs MC vs DP, bootstrapping, bias–variance, on/off-policy, Robbins–Monro, deadly triad |

## Setup used in the experiments

- **State discretisation:** 20 × 20 bins over position `[-1.2, 0.6]` and velocity `[-0.07, 0.07]` (400 states); a 40 × 40 grid is also tested
- **Action discretisation:** 15 equally spaced forces in `[-1, 1]`
- **Reward shaping:** potential-based, `Φ(s) = 4 · sin(3 · position)`
- **Hyper-parameters:** γ = 0.99, α = 0.5, ε decays from 0.5 to 0.01 (decay 0.995 per episode), 2000 episodes, seed 42

## Results (average reward over the last 100 episodes)

| Experiment | Setting | Avg reward |
|---|---|---|
| SARSA vs Q-Learning | SARSA | 37.57 |
| | Q-Learning | 101.87 |
| Learning rate (Q-Learning) | α = 0.1 | 32.08 |
| | α = 0.5 | 101.87 |
| | α = 0.9 | 103.22 |
| Exploration (fixed ε, Q-Learning) | ε = 0.01 | 100.76 |
| | ε = 0.10 | 23.43 |
| | ε = 0.30 | 90.64 |
| Discretisation (Q-Learning) | 20 × 20 (33.7 s) | 101.87 |
| | 40 × 40 (78.1 s) | 26.29 |

Single run with seed 42 — these numbers come from the saved notebook outputs and will vary with a different seed or library version.



> **Python 3.12 or newer is required.** One cell (section 3.5) uses nested quotes inside an f-string, which is only valid syntax from Python 3.12.

Running all cells takes a few minutes (each 2000-episode training run takes roughly 30–80 s).


## Tech stack

Python · NumPy · Matplotlib · Gymnasium
