import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def train_and_evaluate(args: argparse.Namespace) -> None:
    train_env = Monitor(gym.make("CartPole-v1"))
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Saved model to: {args.model_path}")

    eval_env = Monitor(gym.make("CartPole-v1"))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Evaluation over {args.eval_episodes} episodes: mean={mean_reward:.2f}, std={std_reward:.2f}")

    render_requested = args.render_demo or args.render_eval or args.record_and_render
    record_requested = args.record_video or args.record_and_render
    if render_requested:
        render_policy(model=model, episodes=args.render_episodes, seed=args.seed)

    if record_requested:
        record_policy_video(model=model, episodes=args.video_episodes, seed=args.seed, video_dir=args.video_dir)


def render_policy(model: PPO, episodes: int, seed: int) -> None:
    render_env = gym.make("CartPole-v1", render_mode="human")
    returns = []

    for i in range(episodes):
        obs, _ = render_env.reset(seed=seed + 40_000 + i)
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
        returns.append(ep_return)
        print(f"Rendered episode {i + 1}: return={ep_return:.1f}")

    avg_return = sum(returns) / max(len(returns), 1)
    print(f"Render evaluation over {episodes} episodes: avg_return={avg_return:.2f}")
    render_env.close()


def record_policy_video(model: PPO, episodes: int, seed: int, video_dir: str) -> None:
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    try:
        video_env = RecordVideo(
            env=base_env,
            video_folder=video_dir,
            episode_trigger=lambda episode_idx: episode_idx < episodes,
            name_prefix="ppo_cartpole",
        )
    except Exception as exc:
        base_env.close()
        print(f"Video recording unavailable: {exc}")
        print("Install dependency with: uv pip install moviepy")
        return

    returns = []
    for i in range(episodes):
        obs, _ = video_env.reset(seed=seed + 70_000 + i)
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
            ep_return += float(reward)

        returns.append(ep_return)

    video_env.close()

    avg_return = sum(returns) / max(len(returns), 1)
    print(f"Saved {episodes} evaluation video episode(s) to '{video_dir}' | avg_return={avg_return:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gymnasium CartPole-v1 solved with PPO (Stable-Baselines3)")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--model-path", type=str, default="models/ppo_cartpole")
    parser.add_argument("--render-eval", action="store_true", help="Render post-training deterministic evaluation")
    parser.add_argument("--render-episodes", type=int, default=3, help="Number of rendered evaluation episodes")
    parser.add_argument("--record-video", action="store_true", help="Record post-training deterministic evaluation video")
    parser.add_argument("--record-and-render", action="store_true", help="Enable both --render-eval and --record-video")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    parser.add_argument("--video-dir", type=str, default="videos/ppo_cartpole", help="Output directory for recorded videos")
    parser.add_argument("--render-demo", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
