# RL Practice — 2D Reinforcement Learning

Two self-contained RL environments implemented from scratch using **PPO (Proximal Policy Optimization)** with PyTorch. No external RL libraries required.

## Environments

### 1. 2D Robot Walking (`robot_walking_rl.py`)
A simplified bipedal robot learns to walk forward by controlling torques on its hip and knee joints.

- **State**: torso angle, torso velocity, 4 joint angles, 2 foot contact flags
- **Actions**: torques on hip1, knee1, hip2, knee2 (continuous, [-1, 1])
- **Reward**: forward progress, torso stability, energy efficiency; penalty for falling
- **Done**: robot falls (torso angle > 60°) or 1000 steps reached

### 2. 2D Drone Navigation (`drone_navigation_rl.py`)
A drone learns to fly from a random start position to a goal while avoiding circular obstacles and wind disturbance.

- **State**: relative position to goal, velocity, distance, angle, obstacle positions
- **Actions**: thrust in x and y directions (continuous, [-1, 1])
- **Reward**: progress toward goal, reaching goal (+200), energy penalty, collision penalty (-100)
- **Done**: goal reached, obstacle collision, boundary hit, or 500 steps reached

## Algorithm: PPO

Both environments use the same PPO implementation:
- Actor-Critic network with shared backbone
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train the robot walker (~200k steps)
python robot_walking_rl.py

# Train the drone navigator (~300k steps)
python drone_navigation_rl.py
```

Each script saves:
- A trained model (`.pth` file)
- A training curve plot (`.png` file)

The drone script also renders a final episode trajectory.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
