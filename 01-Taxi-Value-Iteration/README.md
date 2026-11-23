# Assignment 1 — Taxi Planning and Model Learning

Original assignment team: Aya Sabry, Alaa Hassan, Shook Mohammed, Ali Elsayed, and Alaa Kurdi.

This notebook explores model-based reinforcement learning through three experiments:

1. **Taxi-v3 with a known model:** Value Iteration and Policy Iteration are implemented and compared across discount factors.
2. **Taxi-v3 with a learned model:** transitions and rewards are estimated from random interaction, then used for planning.
3. **MountainCarContinuous-v0:** continuous states and actions are discretized, an approximate model is learned from interaction, and tabular planning is applied.

## Algorithms and concepts

- Bellman optimality backups
- Value Iteration
- Policy Iteration (policy evaluation and improvement)
- Empirical transition-model estimation
- State/action discretization
- Model approximation and sampling limitations

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Open `Taxi_Value_Iteration_and_Model_Learning.ipynb` in Jupyter or Google Colab and run the cells from top to bottom.

## Environments

- `Taxi-v3` (Gymnasium)
- `MountainCarContinuous-v0` (Gymnasium)

No external dataset is required. The notebook collects its own experience by interacting with the environments.

## Notes on interpretation

The Taxi experiment with the built-in transition table is a planning problem with a known model. The later Taxi and MountainCar experiments estimate a model from sampled transitions. Their results are approximate and depend on exploration coverage. For unseen state-action pairs, the notebook uses an explicitly documented pessimistic self-loop fallback; this is a practical assumption, not a true environment transition.

The MountainCar experiment is a tabular approximation of a continuous-control task. Discretization and action quantization limit the policy's ability to represent the original problem exactly.


