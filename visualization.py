"""Matplotlib learning-curve generation for the maze agent."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plot(history, output_path):
    episodes = [result.episode for result in history]
    rewards = [result.reward for result in history]
    steps = [result.steps for result in history]
    epsilons = [result.epsilon for result in history]

    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    figure.patch.set_facecolor("#181d26")
    series = (
        (rewards, "Episode reward", "#2ecc71"),
        (steps, "Steps", "#3498db"),
        (epsilons, "Epsilon", "#f1c40f"),
    )
    for axis, (values, label, color) in zip(axes, series):
        axis.set_facecolor("#202733")
        axis.plot(episodes, values, color=color, linewidth=1.2)
        axis.set_ylabel(label, color="#e9edf2")
        axis.tick_params(colors="#aeb7c4")
        axis.grid(alpha=0.15)
        for spine in axis.spines.values():
            spine.set_color("#46505f")

    axes[-1].set_xlabel("Episode", color="#e9edf2")
    figure.suptitle("Q-Learning Progress", color="white", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path
