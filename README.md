# AI for AC Optimal Power Flow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning approach to solving AC Optimal Power Flow (AC-OPF) problems using neural networks. This implementation delivers fast, scalable solutions for power system optimization by approximating the complex, nonlinear relationships in electrical power grids.

## Overview

AC Optimal Power Flow is a key optimization problem in power system operations that determines the optimal generator settings while minimizing costs and satisfying operational constraints. Traditional optimization methods often struggle with the scale and complexity of modern power grids. This project uses neural networks to rapidly approximate AC-OPF solutions.

## Project structure

- [optimal_power_flow.ipynb](./optimal_power_flow.ipynb) – notebook with an implementation of the PJM 5 network, including data preprocessing, NN training, evaluation, and comparison of both networks.
- [intro_and_theory.md](./intro_and_theory.md) – in-depth description of the problem, its significance, and the challenges of traditional solutions.
- [assets](./assets/) – folder with images used for explanations.
