"""Train and evaluate the Q-learning maze agent."""

from dataclasses import dataclass

from agent import QLearningAgent
from environment import MazeEnvironment


@dataclass
class EpisodeResult:
    episode: int
    reward: int
    steps: int
    epsilon: float
    reached_goal: bool


def run_episode(env, agent, max_steps=250, training=True):
    """Run one episode and return its reward, steps, and outcome."""
    state = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        action = agent.choose_action(state, training=training)
        next_state, reward, done = env.step(action)

        if training:
            agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward, env.steps, env.done


def train_agent(episodes=1000, max_steps=250, seed=42, report_every=100):
    """Train an agent and return it together with per-episode history."""
    env = MazeEnvironment()
    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        seed=seed,
    )
    history = []

    for episode in range(1, episodes + 1):
        reward, steps, reached_goal = run_episode(
            env, agent, max_steps=max_steps, training=True
        )
        history.append(
            EpisodeResult(
                episode=episode,
                reward=reward,
                steps=steps,
                epsilon=agent.epsilon,
                reached_goal=reached_goal,
            )
        )
        agent.decay_exploration()

        if report_every and (episode == 1 or episode % report_every == 0):
            window = history[-min(report_every, len(history)) :]
            successes = sum(result.reached_goal for result in window)
            average_steps = sum(result.steps for result in window) / len(window)
            average_reward = sum(result.reward for result in window) / len(window)
            print(
                f"Episode {episode:4d} | "
                f"avg reward {average_reward:7.1f} | "
                f"avg steps {average_steps:6.1f} | "
                f"success {successes / len(window):6.1%} | "
                f"epsilon {window[-1].epsilon:.3f}"
            )

    return agent, history


def evaluate_agent(agent, episodes=20, max_steps=250):
    """Evaluate the greedy policy without exploration or Q-table updates."""
    env = MazeEnvironment()
    results = [
        run_episode(env, agent, max_steps=max_steps, training=False)
        for _ in range(episodes)
    ]
    successes = sum(reached_goal for _, _, reached_goal in results)
    successful_steps = [steps for _, steps, reached_goal in results if reached_goal]

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "average_steps": (
            sum(successful_steps) / len(successful_steps)
            if successful_steps
            else None
        ),
    }


def main():
    print("Training Q-learning agent...\n")
    agent, _ = train_agent()
    evaluation = evaluate_agent(agent)

    average_steps = evaluation["average_steps"]
    steps_text = f"{average_steps:.1f}" if average_steps is not None else "N/A"
    print("\nGreedy-policy evaluation")
    print(f"Success: {evaluation['successes']}/{evaluation['episodes']}")
    print(f"Success rate: {evaluation['success_rate']:.1%}")
    print(f"Average successful steps: {steps_text}")


if __name__ == "__main__":
    main()
