# Q-Learning Maze Agent

A Python reinforcement-learning project where you can solve a maze manually,
train a tabular Q-learning agent, and watch its learned policy escape.

## Features

- Manual Pygame maze with collision detection and move counting
- Small Gym-style environment with `reset()` and `step(action)`
- NumPy Q-table and epsilon-greedy action selection
- Reward, step, success-rate, and epsilon tracking
- Animated trained-agent playback with path visualization
- Matplotlib learning curves
- Saved Q-table that is restored on the next launch

## Setup

```powershell
cd "C:\Users\moura\OneDrive\Desktop\mini-games\Q-learning"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Controls

| Mode | Control | Action |
|---|---|---|
| Menu | Mouse | Select play, train, watch, or graphs |
| Manual | Arrow keys | Move the player |
| Manual | R | Reset the maze |
| Watch | R | Replay the learned route |
| Watch | + / - | Increase or decrease animation speed |
| Any mode | Esc | Return to the menu |

## Reinforcement-learning model

The state is the agent's `(row, column)` position. Actions are up, down, left,
and right. Normal movement gives `-1`, hitting a wall gives `-5`, and reaching
the goal gives `+100`.

The agent updates its table with:

```text
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
```

Run training without the graphical interface using:

```powershell
python training.py
```
