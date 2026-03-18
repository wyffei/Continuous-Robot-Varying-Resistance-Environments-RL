# Adaptive Continuous Robot Control in Varying Resistance Environments Using Reinforcement Learning

This project studies reinforcement-learning-based control of a continuous multi-link robotic arm operating in environments with spatially varying resistance, with a particular focus on air–water interface scenarios.

The goal is to train a control policy that enables the robot arm to reach target positions accurately while adapting to changing drag and hydrodynamic effects, and at the same time reducing total energy consumption and motion time.

## Overview

Robotic motion across different media, such as air and water, introduces abrupt changes in resistance, drag, and added-mass effects. These discontinuities make conventional controllers difficult to apply reliably. This project builds a full reinforcement learning framework in MuJoCo to address this problem.

The system models a 10-DOF planar robotic arm moving in a two-fluid environment. During motion, different arm segments may lie in different media, resulting in time-varying hydrodynamic loads along the robot body. The controller must adapt online to these varying resistive forces while completing a target-reaching task efficiently. 

## Project Goals

- Build a MuJoCo-based simulation environment for a multi-link robot arm in a two-medium setting
- Model varying fluid resistance across an air–water interface
- Train a reinforcement learning agent for adaptive target reaching
- Compare RL performance with inverse kinematics (IK) baselines
- Evaluate control quality in terms of accuracy, stability, and energy consumption

## Problem Setting

### Robot

- 10-DOF planar serial manipulator
- Operates in the x-z plane
- End-effector target reaching task in Cartesian space

### Environment

The workspace is divided into two horizontal fluid regions:

- **Fluid 1 (water)**: below the interface
- **Fluid 2 (air)**: above the interface

A smooth transition function is used near the interface to avoid discontinuities in the fluid parameters. At every simulation step, each arm segment is assigned effective hydrodynamic parameters based on its position relative to the interface.

### Task

At the beginning of each episode, a target position is sampled from a reachable workspace. The controller must move the end-effector to the target within a fixed horizon while minimizing:

- final reaching error
- total energy consumption
- excessive collision and unstable motion

## Method

## Reinforcement Learning Framework

The RL framework contains the following components:

### Observation Space

The observation vector includes:

- joint positions
- joint velocities
- end-effector Cartesian position
- end-effector position error relative to the target
- remaining episode time

This combines both robot-state information and task-state information for policy learning.

### Action Space

The final controller uses:

- **10-dimensional desired joint velocities**

The commanded velocities are integrated inside the environment and tracked by a built-in PID controller, which produces the final joint torques applied in MuJoCo.

Compared with direct torque control, velocity-based action design improves training stability and results in smoother motions.

### Reward Function

The reward includes several terms:

- task progress reward
- success bonus
- energy penalty
- collision penalty
- time reward

This reward design encourages the agent to reach the target accurately while avoiding wasteful motion and reducing total energy consumption.

## Learning Agent

The training agent is based on PPO / recurrent PPO with:

- MLP feature extractor: `[256, 256, 128]`
- LSTM layer: 128 units
- ReLU activations
- entropy decay during training

### Training Hyperparameters

- learning rate: `2e-4`
- discount factor: `0.995`
- GAE lambda: `0.95`
- PPO clip range: `0.2`
- batch size: `128`
- rollout length: `256`
- total training steps: `800000`

## Neural Network

The policy and value networks are dual-branch MLPs with the same hidden-layer layout:

- Linear → 256 → ReLU
- Linear → 256 → ReLU
- Linear → 128 → ReLU

The actor outputs the mean action vector, while a learnable `log_std` controls action variance.  
The critic outputs a scalar state-value estimate.

## Dependencies

This project is built on the following libraries:

- Python 3.x
- MuJoCo
- NumPy
- SciPy
- PyTorch
- Stable-Baselines3
- Gymnasium
- Matplotlib

Example installation:

```bash
pip install numpy scipy matplotlib torch gymnasium stable-baselines3
