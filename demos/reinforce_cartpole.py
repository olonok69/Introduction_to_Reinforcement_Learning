import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.model(obs)
        return torch.softmax(logits, dim=-1)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
    return returns_tensor


def train(args: argparse.Namespace) -> None:
    render_requested = args.render_eval or args.record_and_render
    record_requested = args.record_video or args.record_and_render

    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = PolicyNetwork(obs_size, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    reward_window = deque(maxlen=100)

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs = policy(obs_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(float(reward))
            done = terminated or truncated

        returns = discounted_returns(rewards, args.gamma)
        policy_loss = []
        for log_prob, ret in zip(log_probs, returns):
            policy_loss.append(-log_prob * ret)
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_reward = sum(rewards)
        reward_window.append(ep_reward)
        avg_100 = float(np.mean(reward_window))

        if episode % args.log_every == 0:
            print(
                f"Episode {episode:4d} | reward={ep_reward:6.1f} | avg100={avg_100:7.2f} | loss={loss.item():8.4f}"
            )

        if avg_100 >= args.solve_score and episode >= 100:
            print(f"Solved in {episode} episodes with avg100={avg_100:.2f}")
            break

    env.close()

    if render_requested:
        render_policy(policy=policy, episodes=args.render_episodes, seed=args.seed)

    if record_requested:
        record_policy_video(
            policy=policy,
            episodes=args.video_episodes,
            seed=args.seed,
            video_dir=args.video_dir,
        )


def render_policy(policy: PolicyNetwork, episodes: int, seed: int) -> None:
    render_env = gym.make("CartPole-v1", render_mode="human")
    returns = []

    policy.eval()
    with torch.no_grad():
        for i in range(episodes):
            obs, _ = render_env.reset(seed=seed + 30_000 + i)
            done = False
            ep_return = 0.0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                probs = policy(obs_tensor)
                action = int(torch.argmax(probs, dim=-1).item())

                obs, reward, terminated, truncated, _ = render_env.step(action)
                done = terminated or truncated
                ep_return += float(reward)

            returns.append(ep_return)
            print(f"Rendered episode {i + 1}: return={ep_return:.1f}")

    avg_return = float(np.mean(returns)) if returns else 0.0
    print(f"Render evaluation over {episodes} episodes: avg_return={avg_return:.2f}")
    render_env.close()


def record_policy_video(policy: PolicyNetwork, episodes: int, seed: int, video_dir: str) -> None:
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    try:
        video_env = RecordVideo(
            env=base_env,
            video_folder=video_dir,
            episode_trigger=lambda episode_idx: episode_idx < episodes,
            name_prefix="reinforce_cartpole",
        )
    except Exception as exc:
        base_env.close()
        print(f"Video recording unavailable: {exc}")
        print("Install dependency with: uv pip install moviepy")
        return

    returns = []
    policy.eval()
    with torch.no_grad():
        for i in range(episodes):
            obs, _ = video_env.reset(seed=seed + 60_000 + i)
            done = False
            ep_return = 0.0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                probs = policy(obs_tensor)
                action = int(torch.argmax(probs, dim=-1).item())

                obs, reward, terminated, truncated, _ = video_env.step(action)
                done = terminated or truncated
                ep_return += float(reward)

            returns.append(ep_return)

    video_env.close()

    avg_return = float(np.mean(returns)) if returns else 0.0
    print(f"Saved {episodes} evaluation video episode(s) to '{video_dir}' | avg_return={avg_return:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE on CartPole-v1 (Gymnasium)")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--solve-score", type=float, default=475.0)
    parser.add_argument("--render-eval", action="store_true", help="Render post-training greedy evaluation")
    parser.add_argument("--render-episodes", type=int, default=3, help="Number of rendered evaluation episodes")
    parser.add_argument("--record-video", action="store_true", help="Record post-training greedy evaluation video")
    parser.add_argument("--record-and-render", action="store_true", help="Enable both --render-eval and --record-video")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    parser.add_argument("--video-dir", type=str, default="videos/reinforce_cartpole", help="Output directory for recorded videos")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
