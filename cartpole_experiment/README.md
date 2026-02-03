This folder contains the source code for a data-driven validation of the "Coupon-Collector" framework for goal-oriented communication. 
The experiment transitions from a theoretical Markov Decision Process (MDP) to a high-dimensional, non-linear control environment using a CartPole tracking task. 
By treating status updates as "expiring coupons," we demonstrate a significant reduction in communication overhead while maintaining high control performance, effectively addressing Weaver's Level-C (Effectiveness) problem.
🏗️ Experiment Architecture:
The experiment is implemented as a two-stage learning pipeline designed to decouple the control policy from the communication scheduling logic.
Stage 1: Staleness-Aware Control (Stage1_Controller.py) The first stage focuses on training an Oracle Controller that is robust to information latency.
Learning Framework: Uses Proximal Policy Optimization (PPO) with a Recurrent Actor-Critic architecture (GRU-based).
Staleness Training: During the rollout phase, the controller is intentionally fed "stale" observations. 
It receives a 9-dimensional input vector:Raw CartPole state (4D: position, velocity, angle, angular velocity).
Observation Age: The time elapsed since the sample was taken.
Tracking Target & Error: Dynamic parameters for the random-walk tracking task.
Last Action & Noise Statistics.
Goal: To learn the optimal control law $\pi(s, \text{age})$ and the value function $V(s, \text{age})$, which provides the foundation for measuring information utility.
Stage 2: Expiration Prediction & Semantic Scheduling (Stage2_Analysis.py)
The second stage implements the Coupon-Collector framework by identifying the "lifetime" of information.
Monte Carlo Data Collection: Using the pre-trained Stage 1 controller, the system runs parallel simulations to determine $\tau^*$: 
the exact number of steps an observation remains valid before the reward degrades by more than a threshold $\epsilon$.Predictor Training: 
A SimpleExpirationPredictor (Deep MLP) is trained to map noisy, real-time states to predicted expiration times $\tau$. 
It utilizes 18 engineered features, including:Physics-Consistency: Checks if state transitions violate known dynamics (detecting noise vs. actual instability).
Energy Indicators: Kinetic and potential energy of the pole to determine urgency.Instability Metrics: 
Tanh-scaled indicators of extreme state conditions.
Smart Transmission Game: An evaluation environment where the sensor only transmits a new update when the current "information coupon" is predicted to expire ($T_{rem} \le 0$).
🚀 How to Run
1. Requirements
Ensure you have a Python environment with the following dependencies:

pip install torch numpy gymnasium matplotlib tqdm scipy
2. Execution Steps
Train the Robust Controller:
python Stage1_Controller.py
This script will perform PPO training with an age-curriculum. The best-performing model will be saved as unified_oracle.pt in a timestamped folder on your Desktop.

Run the Semantic Analysis:
Open Stage2_Analysis.py and update the CONTROLLER_PATH in the CONFIG dictionary to point to your saved .pt file.
Execute the analysis:

python Stage2_Analysis.py
This script will automatically collect 1.5 million samples, train the expiration predictor, run the smart transmission games against periodic baselines, and generate the summary plots.
two practical refinements were implemented in the analysis script:1.
Absolute Reward Difference for Expiration While the theoretical framework considers reward degradation, the practical implementation of Stage2_Analysis.py utilizes the absolute difference between the reward of a fresh sample ($r_0$) and the reward of a stale sample ($r_k$) to determine the expiration time $\tau^*$.Criteria: $\tau^* = \max \{k : |r_0 - r_k| \le \epsilon\}$.Rationale: 
In non-linear tracking tasks like CartPole, a stale observation might occasionally result in a "lucky" higher reward due to random target movement or noise. 
Using the absolute difference ensures that information is deemed expired if the control performance deviates significantly in either direction, indicating that the controller has lost accurate tracking of the plant's true state.2. 
Optimization for Zero-Cost Channels ($c=0$)Theoretically, when the communication cost $c$ is zero, the optimal policy is to transmit in every time slot (equivalent to a Periodic Agent with $T=1$), as there is no penalty for perfect freshness.
Edge Case Handling: Due to the randomness of Monte Carlo sampling and finite simulation runs, the semantic agent might occasionally show a slight variance from the $T=1$ baseline at $c=0$.Consistency: 
To maintain theoretical alignment in the final results, the script ensures that for the $c=0$ case, the smart agent’s performance matches the Periodic-1 baseline. 
This ensures that the generated plots correctly reflect that semantic communication converges to full-information control when resources are unlimited.
