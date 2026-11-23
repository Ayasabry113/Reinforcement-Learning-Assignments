# Blackjack — Monte Carlo Reinforcement Learning

This project implements and compares Monte Carlo (MC) methods in Gymnasium's `Blackjack-v1` environment, then explores the limitations of applying tabular MC to `MountainCarContinuous-v0` after discretising its state and action spaces.

## Contents

- **OOP environment wrapper** for Gymnasium Blackjack.
- **MC Prediction:** First-Visit and Every-Visit policy evaluation, plus a comparison of incremental and batch averaging.
- **On-Policy MC Control:** epsilon-greedy exploration with several epsilon values.
- **Off-Policy Evaluation:** ordinary and weighted importance sampling for a deterministic target policy.
- **MountainCar extension:** state/action discretisation, tabular MC control, prediction, and discussion of limitations.
- **Conceptual questions** covering MC theory and its relationship to Dynamic Programming.

## Run

The notebook is designed for Google Colab or a local Jupyter environment.

```bash
python -m pip install -r requirements.txt
jupyter notebook Blackjack_Monte_Carlo.ipynb
```

Run the notebook from top to bottom. The experiments use 100,000 episodes for several Blackjack comparisons and may take some time depending on your machine.

## Environments and action encoding

- `Blackjack-v1`: action `0 = stick`, action `1 = hit`.
- `MountainCarContinuous-v0`: the continuous action is approximated using five discrete force values `[-1.0, -0.5, 0.0, 0.5, 1.0]`; position and velocity are discretised into 20 bins each.

## Important implementation notes

- First-Visit MC updates the first occurrence of a state (or state-action pair) in chronological episode order. Returns are computed backwards, then updates are applied forwards to preserve the correct First-Visit definition.
- Incremental and batch averaging compute the same arithmetic mean; incremental averaging avoids storing the cumulative sum of all returns.
- The MountainCar experiment is an educational extension, not a claim of solving the environment. The notebook discusses limited exploration, discretisation, and the sample-efficiency limitations of tabular MC.

## Reproducibility

The notebook sets NumPy's random seed to `42` in the setup cell and resets it before the main experiments. Gymnasium environments are stochastic, so exact results can vary across library versions.

