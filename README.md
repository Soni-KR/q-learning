# Velocity Maze: Human vs Q-Learner

A speed-racing-inspired Python reinforcement-learning game where a human races
an agent that learns the same maze live. The visual theme is an original homage
to fast 1990s platform games, using vector shapes rather than copied assets.

## Features

- Manual Pygame maze with collision detection and move counting
- Small Gym-style environment with `reset()` and `step(action)`
- NumPy Q-table and epsilon-greedy action selection
- Reward, step, success-rate, and epsilon tracking
- Animated trained-agent playback with path visualization
- Matplotlib learning curves
- Saved Q-table that is restored on the next launch
- Three circuits with 11-, 24-, and 58-move optimal routes
- Human vs learner mode: each human tile funds live AI training
- Separate saved model and learning telemetry for every circuit
- Blue human and orange AI racers with speed trails and ring-like goals
- Sonic-reference app icon and menu badge with blue quills, speed lines, and a ring

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
| Menu | Circuit button | Cycle through the three mazes |
| Menu | Mouse | Select time trial, live race, training, replay, or telemetry |
| Manual | Arrow keys | Move the player |
| Manual | R | Reset the maze |
| Race | Arrow keys | Move and grant the AI its next training budget |
| Race | R | Start a fresh rematch with an empty Q-table |
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
