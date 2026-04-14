# PBRL-EMOA

This repository contains preference-based multi-objective optimization experiments built around a DDPG-assisted EMOA workflow.

## Project Layout

```text
pbrl_emoa_project/
├── configs/
│   └── experiments/
│       └── journal_params.csv
├── scripts/
│   └── run_experiment.py
├── src/
│   └── pbrl_emoa/
│       ├── algorithms/
│       │   └── pbrl_emoa.py
│       ├── operators/
│       │   └── ea_operators.py
│       ├── preference/
│       │   ├── machine_dm.py
│       │   ├── rank_svm.py
│       │   └── value_functions.py
│       ├── problems/
│       │   └── core.py
│       └── rl/
│           └── ddpg_agent.py
└── tests/
    ├── test_imports.py
    └── test_value_functions.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run an experiment

```bash
python scripts/run_experiment.py <idi> <test> <mode> <interactions> <run> <g> <s> <q> <factor>
```

Example:

```bash
python scripts/run_experiment.py 1 1 3 10 1 - - 0 nobias
```

## Notes

- `tests.py` from the original repository was an experiment runner, not a unit-test module, so it has been moved to `scripts/run_experiment.py`.
- `ddpgemoa.py` and `ddpgrank.py` had a circular dependency risk; the RL agent was separated into `rl/ddpg_agent.py`, and the main algorithm now imports it in one direction only.
- `journal_params.csv` is stored under `configs/experiments/`.
