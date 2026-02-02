# Coupon-Collector Q-Learning

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Structure-Aware Q-Learning for optimal scheduling in coupon-collector systems with expiring samples.

## Problem Overview

A sender receives fresh data samples with random lifetimes and must decide when to transmit them over an unreliable channel. The receiver earns rewards for holding valid (unexpired) samples.

| Parameter | Description |
|-----------|-------------|
| `K` | Maximum sample lifetime (TTL) |
| `r` | Reward per slot with valid sample at receiver |
| `c` | Transmission cost |
| `p_s` | Channel success probability |
| `p_T` | Distribution of fresh sample lifetimes |

**Objective**: Maximize long-run average reward ρ = E[reward] - E[cost]

## Key Theoretical Results

1. **Threshold Structure**: Optimal policy transmits iff T_r ≤ θ(T_s)
2. **Monotonicity**: θ(T_s) ≤ θ(T_s + 1) for all T_s
3. **Structural Constraint**: Never transmit when T_r > T_s (obsolete data)

## Quick Start

### Requirements
- MATLAB R2020a or later
- No additional toolboxes

### Run

```matlab
% In MATLAB, navigate to folder and run:
compare_q_learning