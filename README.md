# Abyss Submarine: DQN-based Autonomous Navigation 🌊

This repository contains "Abyss Submarine," a continuous-state Reinforcement Learning environment where an Autonomous Underwater Vehicle (AUV) learns to navigate a hazardous marine environment using a Deep Q-Network (DQN). 

## 🚀 Project Highlights
* **Algorithm**: Deep Q-Network (DQN) implemented in PyTorch.
* **Environment**: Custom continuous Markov Decision Process (MDP) built with Pygame.
* **State Space**: 42-dimensional continuous vector capturing real-time spatial dynamics.
* **Innovation**: Implemented Prior Reward Shaping using point-to-line trajectory evaluation to solve sparse reward issues.
* **Physics Integration**: Simulates real-world fluid dynamics with a continuous lateral ocean current constraint.

## 📁 Repository Structure
* `main.py` - The main simulation loop and Pygame rendering engine.

## 🎮 Demo Video
Watch the trained AI agent actively dodging dynamic obstacles here:
**[Insert your YouTube Link Here]**

## 🛠️ Tech Stack
* Python 3.x
* PyTorch
* Pygame
