"""A tabular Q-learning agent implemented with NumPy."""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        rows,
        cols,
        action_count=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        seed=None,
    ):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0 <= min_epsilon <= epsilon <= 1:
            raise ValueError("epsilon values must satisfy 0 <= min_epsilon <= epsilon <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")

        # q_table[row, column, action] stores Q(state, action).
        self.q_table = np.zeros((rows, cols, action_count), dtype=np.float64)
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, training=True):
        """Choose an action using epsilon-greedy exploration."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))

        values = self.q_table[state]
        best_actions = np.flatnonzero(values == values.max())
        # Random tie-breaking avoids always preferring UP when values are equal.
        return int(self.rng.choice(best_actions))

    def learn(self, state, action, reward, next_state, done):
        """Apply one Bellman/Q-learning update and return the new Q-value."""
        current_q = self.q_table[state][action]
        future_q = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.discount_factor * future_q

        updated_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state][action] = updated_q
        return float(updated_q)

    def decay_exploration(self):
        """Reduce exploration once at the end of each training episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

