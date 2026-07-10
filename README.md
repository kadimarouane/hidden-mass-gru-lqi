# Hidden-Mass GRU-LQI Benchmark

This repository contains a Python benchmark for robust tracking control of a mass-spring-damper system with hidden and time-varying mass.

## System model

The studied system is:

m(t) x_ddot(t) + b x_dot(t) + k x(t) = u(t)

where:

- m(t) is the time-varying mass,
- b is the damping coefficient,
- k is the spring stiffness,
- u(t) is the control force.

## Controllers

The benchmark compares four controllers:

1. Fixed LQI designed at a nominal mass
2. True-mass gain-scheduled LQI used as an oracle reference
3. RLS estimated-mass gain-scheduled LQI
4. GRU hidden-mass neural policy

## Main idea

The GRU policy receives only measured history and previous control actions.  
The true mass is not provided to the GRU controller.

## Installation

Install the required Python packages:

pip install numpy scipy pandas matplotlib torch

## Run

To run the benchmark:

python hidden_mass_gru_lqi.py

## Outputs

The script generates:

- metrics tables,
- statistical summaries,
- trained models,
- PDF, SVG, and PNG figures.

The outputs are saved in the results folder.

## Author

Kadi Marouane
