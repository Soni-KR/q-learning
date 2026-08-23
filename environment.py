"""The maze exposed as a small reinforcement-learning environment."""

from maze import GOAL_POSITION, MAZE, START_POSITION, is_walkable


class MazeEnvironment:
    """A minimal Gym-like environment with reset() and step()."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = (UP, DOWN, LEFT, RIGHT)
    ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
    ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    MOVE_REWARD = -1
    WALL_REWARD = -5
    GOAL_REWARD = 100

    def __init__(self):
        self.rows = len(MAZE)
        self.cols = len(MAZE[0])
        self.position = START_POSITION
        self.done = False
        self.steps = 0

    @property
    def state(self):
        """The current state is simply the agent's (row, column)."""
        return self.position

    def reset(self):
        """Start a new episode and return its initial state."""
        self.position = START_POSITION
        self.done = False
        self.steps = 0
        return self.state

    def step(self, action):
        """Apply one action and return (next_state, reward, done)."""
        if action not in self.ACTIONS:
            raise ValueError(f"Action must be one of {self.ACTIONS}; received {action!r}")
        if self.done:
            raise RuntimeError("Episode is finished; call reset() before step()")

        row_change, col_change = self.ACTION_DELTAS[action]
        row, col = self.position
        candidate = row + row_change, col + col_change
        self.steps += 1

        if not is_walkable(*candidate):
            # A wall hit costs more and leaves the agent in the same state.
            reward = self.WALL_REWARD
        else:
            self.position = candidate
            reward = self.MOVE_REWARD

        if self.position == GOAL_POSITION:
            reward = self.GOAL_REWARD
            self.done = True

        return self.state, reward, self.done

