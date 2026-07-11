# Hidden-Mass GRU-LQI Benchmark

This repository contains the Python implementation of a comparative
benchmark for controlling a mass-spring-damper system with hidden and
time-varying mass.

## Controllers

The benchmark compares four controllers:

1. Fixed LQI designed for the nominal mass
2. True-mass gain-scheduled LQI used as an oracle reference
3. RLS estimated-mass gain-scheduled LQI
4. GRU hidden-mass learning policy

## Main file

```text
hidden_mass_gru_lqi.py
