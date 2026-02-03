# Beyond Freshness and Semantics: A Coupon-Collector Framework for Effective Status Updates

This repository contains the extended technical report and implementation code for the paper "Beyond Freshness and Semantics: A Coupon-Collector Framework for Effective Status Updates."

## 📄 Abstract
[cite_start]For status update systems operating over unreliable energy-constrained wireless channels, we address Weaver's Level-C question: do my packets actually improve the plant's behaviour?[cite: 8]. [cite_start]We cast the problem as a coupon-collector variant with expiring coupons, prove that the optimal schedule is doubly thresholded, and design a Structure-Aware Q-learning algorithm (SAQ)[cite: 10]. [cite_start]Simulations show that SAQ matches Value Iteration performance while converging significantly faster than baseline Q-learning[cite: 11].

## 📂 Repository Structure
* **`Beyond_Freshness_Extended.pdf`**: The full version of the paper including the detailed proofs for Theorems 1-4 and Lemma 1.
* [cite_start]**`src/`**: Python implementation of the Structure-Aware Q-learning (SAQ) algorithm[cite: 67].
* [cite_start]**`simulations/`**: Code for the CartPole balancing task and the linear Kalman-controlled plant case studies[cite: 63].

## 🎓 Proofs and Mathematical Derivations
The following proofs, omitted from the main conference paper for space, are available in the extended PDF:
* [cite_start]**Theorem 1**: Optimality of the Just-In-Time (JIT) policy[cite: 113, 511].
* [cite_start]**Lemma 1**: Monotonicity of the value function[cite: 331, 617].
* [cite_start]**Theorem 2**: Double-threshold structure of the optimal policy[cite: 338, 638].
* [cite_start]**Theorem 3**: General properties of the optimal policy[cite: 344, 668].
* [cite_start]**Theorem 4**: Closed-form threshold rule for constant lifetimes[cite: 364, 685].
