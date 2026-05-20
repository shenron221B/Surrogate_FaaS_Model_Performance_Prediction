# Machine Learning and Fine-Tuning for Performance Modeling of Function-as-a-Service Systems in Edge Computing Environments

This repository contains the simulation framework, the data extraction scripts, and the Machine Learning pipeline developed for the performance analysis and modeling of the Serverledge Edge FaaS (Function-as-a-Service) platform.

## Overview

As Serverless computing extends towards the Edge, accurately modeling performance and resource allocation in highly constrained environments becomes a critical challenge. Classic analytical models derived from queueing theory (such as M/M/c/K, Erlang Loss, and advanced Multiserver Job FCFS queues) rely on strict mathematical assumptions. While mathematically elegant, they fail to capture the complex, low-level dynamics of physical systems, such as Cold Start delays, Docker orchestration overheads, kernel interference, and CPU thrashing under heavy load. 

To bypass these analytical limitations, discrete-event simulators can be used. However, simulators inherently suffer from the "Sim-to-Real gap," as they often underestimate system degradation by relying on idealized execution times and isolated resource modeling.

To overcome these challenges, this project implements a **hybrid, data-driven methodology** that bridges the gap between simulation and reality:
1. **Synthetic Data Generation:** A custom Go-based discrete-event simulator is used to rapidly generate a massive synthetic dataset (thousands of load scenarios), mapping the fundamental queueing theory dynamics of the FaaS node.
2. **Zero-Shot Learning:** A suite of Neural Networks (Multi-Layer Perceptrons) is pre-trained exclusively on this synthetic data, acting as a baseline that understands the theoretical "physics" of the system.
3. **Incremental Fine-Tuning:** To close the Sim-to-Real gap, the pre-trained networks undergo a Transfer Learning phase. By freezing the initial layers to prevent catastrophic forgetting, the models are fine-tuned using a strictly minimal and balanced set of real-world profiling data (ranging from 5 up to 25 real samples). 

The predictive accuracy of this hybrid approach is extensively benchmarked against the pure simulator and state-of-the-art analytical models (including Grosof's asymptotic approach and Kaufman-Roberts), proving its superiority in mapping complex QoS metrics.

## Repository Structure

* **`serverledge/`**: Contains the scripts and utilities to interact with the real Serverledge deployment.
* **`src/`**: Contains the core source code of the project. It includes the discrete-event simulator, the mathematical baseline evaluations, data preprocessing scripts (including polynomial feature engineering), and the complete Machine Learning pipeline handling both the Zero-Shot training and the Fine-Tuning processes.

## Conclusions

The experimental results demonstrate that the proposed Fine-Tuning approach successfully and efficiently corrects the structural biases of the simulator. The Zero-Shot models, while capturing the general trend, exhibit significant errors under saturation. However, the injection of as few as 5 to 25 real-world data points is sufficient to drastically abate prediction errors across all key metrics (Average Response Time, Utility, Success Rate and Cold Start Rate). 

The Fine-Tuned surrogate models effectively learn to compensate for unsimulated physical overheads—such as memory thrashing and CPU contention—providing a highly accurate and lightweight performance oracle at a fraction of the cost required for exhaustive real-world profiling.
