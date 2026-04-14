# PBRL-EMOA

This repository contains preference-based multi-objective optimization experiments built around a DDPG-assisted EMOA workflow for the paper “Agile Project Portfolio Realignment: A Framework of PbRL-EMOA for Shifting Strategic Priority”.

## Run an experiment

```bash
python scripts/run_experiment.py <idi> <test> <mode> <interactions> <run> <g> <s> <q> <factor>
```

Example:

```bash
python scripts/run_experiment.py 1 1 3 10 1 - - 0 nobias
```
