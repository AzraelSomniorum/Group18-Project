# Project Overview
The objective of this project was to develop a Reinforcement Learning (RL) agent capable of solving the classic Inverted Pendulum control theory problem.

The Inverted Pendulum problem system consists of a pendulum that starts in a random position with one end attached to a fixed point, and the other one being free. The goal that the agent must achieve is to apply torque on the free end to swing it into a balanced upright position.

# How to Run
```bash
# Train the agent
python train_pendulum.py --train --episodes 2000

# Render and visualize performance
python train_pendulum.py --render --episodes 10
```

# Project Dependencies
```
Package              Version
-------------------- -------
matplotlib           3.10.0
gymnasium            1.2.2  
numpy                2.2.0
pygame               2.6.1
```


# Contribution List
- Agent Development -> B123040040 余家睿
- Environment Development -> B123040037 陳聖繹
- Training Development, Reflection Report -> B123040061 陳偉財

---------------------------------------------------
# Group Project Setup Guide

## Project Content
- Gymnasium v1.2.2
- Part1 Sample Code
- Part2 Sample Code
- Part3 Sample Code
  
## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib
```

---

## ✅ Verification

Run the following command to verify that the installation is successful:

```bash
% pip list
```

Sample Output from MacOS:

```
Package              Version Editable project location
-------------------- ------- --------------------------------------------
cloudpickle          3.1.2
Farama-Notifications 0.0.4
gymnasium            1.2.2   ./group_project/Gymnasium
numpy                2.3.5
pip                  24.3.1
typing_extensions    4.15.0
```

If your output matches the above (or is similar), your environment is correctly configured.

---

## 🚀 Running the Project

### **Part 1: Mountain Car**
Train and test the reinforcement learning agent:

```bash
# Train the agent
python mountain_car.py --train --episodes 5000

# Render and visualize performance
python mountain_car.py --render --episodes 10
```

### **Part 2: Frozen Lake**
Run the Frozen Lake environment:

```bash
python frozen_lake.py
```

### **Part 3: OOP Project Environment**
Execute the custom OOP environment:

```bash
python oop_project_env.py
```

**Tip:**  
If you’re on Windows, replace  
```bash
source .venv/bin/activate
```  
with  
```bash
.venv\Scripts\activate
```
