import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo


def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(q_table.shape[1])
    return int(np.argmax(q_table[state]))


def train(args: argparse.Namespace) -> None:
    render_requested = args.render_eval or args.record_and_render
    record_requested = args.record_video or args.record_and_render

    env = gym.make("FrozenLake-v1", is_slippery=args.slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    np.random.seed(args.seed)
    q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = args.epsilon_start
    reward_history = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0

        for _ in range(args.max_steps):
            action = epsilon_greedy_action(q_table, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(q_table[next_state])
            td_target = reward + args.gamma * best_next * (1.0 - float(done))
            td_error = td_target - q_table[state, action]
            q_table[state, action] += args.alpha * td_error

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        reward_history.append(total_reward)

        if episode % args.log_every == 0:
            avg_100 = np.mean(reward_history[-100:])
            print(
                f"Episode {episode:5d} | epsilon={epsilon:0.4f} | reward={total_reward:0.1f} | avg100={avg_100:0.3f}"
            )

    print("\nFinal Q-table:")
    np.set_printoptions(precision=3, suppress=True)
    print(q_table)

    eval_score = evaluate_policy(env, q_table, episodes=args.eval_episodes, seed=args.seed)
    print(f"Evaluation success rate over {args.eval_episodes} episodes: {eval_score:0.3f}")
    env.close()

    if render_requested:
        render_policy(
            q_table=q_table,
            episodes=args.render_episodes,
            slippery=args.slippery,
            seed=args.seed,
        )

    if record_requested:
        record_policy_video(
            q_table=q_table,
            episodes=args.video_episodes,
            slippery=args.slippery,
            seed=args.seed,
            video_dir=args.video_dir,
        )


def evaluate_policy(env: gym.Env, q_table: np.ndarray, episodes: int, seed: int) -> float:
    successes = 0
    for i in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done and reward > 0:
                successes += 1
    return successes / episodes


def render_policy(q_table: np.ndarray, episodes: int, slippery: bool, seed: int) -> None:
    render_env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="human")
    successes = 0
    returns = []

    for i in range(episodes):
        state, _ = render_env.reset(seed=seed + 20_000 + i)
        done = False
        ep_return = 0.0

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        if ep_return > 0:
            successes += 1

    avg_return = float(np.mean(returns)) if returns else 0.0
    print(
        f"Render evaluation over {episodes} episodes: success_rate={successes / max(episodes, 1):0.3f}, avg_return={avg_return:0.3f}"
    )
    render_env.close()


def record_policy_video(
    q_table: np.ndarray,
    episodes: int,
    slippery: bool,
    seed: int,
    video_dir: str,
) -> None:
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    base_env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="rgb_array")
    try:
        video_env = RecordVideo(
            env=base_env,
            video_folder=video_dir,
            episode_trigger=lambda episode_idx: episode_idx < episodes,
            name_prefix="q_learning_frozenlake",
        )
    except Exception as exc:
        base_env.close()
        print(f"Video recording unavailable: {exc}")
        print("Install dependency with: uv pip install moviepy")
        return

    successes = 0
    returns = []

    for i in range(episodes):
        state, _ = video_env.reset(seed=seed + 50_000 + i)
        done = False
        ep_return = 0.0

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        if ep_return > 0:
            successes += 1

    video_env.close()

    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = successes / max(episodes, 1)
    print(
        f"Saved {episodes} evaluation video episode(s) to '{video_dir}' | success_rate={success_rate:0.3f}, avg_return={avg_return:0.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabular Q-learning on FrozenLake-v1 (Gymnasium)")
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--slippery", action="store_true", help="Use stochastic transitions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--render-eval", action="store_true", help="Render post-training greedy evaluation")
    parser.add_argument("--render-episodes", type=int, default=3, help="Number of rendered evaluation episodes")
    parser.add_argument("--record-video", action="store_true", help="Record post-training greedy evaluation video")
    parser.add_argument("--record-and-render", action="store_true", help="Enable both --render-eval and --record-video")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    parser.add_argument("--video-dir", type=str, default="videos/q_learning_frozenlake", help="Output directory for recorded videos")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
