import gymnasium as gym
import utils
from agent import DiscretePPOAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

environments = [
    "ALE/Asteroids-v5",
    "ALE/Breakout-v5",
    "ALE/BeamRider-v5",
    "ALE/Centipede-v5",
    "ALE/DonkeyKong-v5",
    "ALE/DoubleDunk-v5",
    "ALE/Frogger-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MarioBros-v5",
    "ALE/MsPacman-v5",
    "ALE/Pong-v5",
    "ALE/Seaquest-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Tetris-v5",
    "ALE/VideoChess-v5",
]


def run_ppo(env_name, n_games=10000):
    env = gym.make(env_name, render_mode="rgb_array")
    print(f"\nEnvironment: {env_name}")
    print(f"Obs.Space: {env.observation_space.shape} Act.Space: {env.action_space.n}")

    agent = DiscretePPOAgent(
        env.observation_space.shape,
        env.action_space.n,
        alpha=3e-5,
        n_epochs=10,
        batch_size=64,
    )

    STEPS = 2048

    n_steps, n_learn, best_score = 0, 0, env.reward_range[0]
    history, metrics = [], []

    for i in range(n_games):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action, prob = agent.choose_action(state)

            next_state, reward, term, trunc, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()

            agent.remember(state, next_state, action, prob, reward, term or trunc)

            n_steps += 1
            if n_steps % STEPS == 0:
                agent.learn()
                n_learn += 1

            score += reward
            state = next_state

            history.append(score)
            avg_score = np.mean(history[-100:])

        if i > 100 and avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        metrics.append(
            {
                "episode": i + 1,
                "average_score": avg_score,
                "best_score": best_score,
            }
        )
        diff = int(abs(avg_score-best_score))
        sign = "+" if diff >=0 else "-"
        print(
            f"[{env_name} Episode {i + 1:04}/{n_games}]   Average Score = {avg_score:7.4f} ({sign}{diff}) ",
            end="\r",
        )

    return history, metrics, best_score, agent


def save_best_version(env_name, agent, seeds=100):
    agent.load_checkpoints()

    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(seeds):
        env = gym.make(env_name, render_mode="rgb_array")

        frames = []
        total_reward = 0

        state, _ = env.reset()
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action, prob = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            state = next_state

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    utils.save_animation(best_frames, f"environments/{env_name}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "-n",
        "--n_games",
        default=10000,
        type=int,
        help="Number of episodes (games) to run during training",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        history, metrics, best_score, trained_agent = run_ppo(args.env, args.n_games)
        utils.plot_running_avg(history, args.env)
        df = pd.DataFrame(metrics)
        df.to_csv(f"metrics/{args.env}_metrics.csv", index=False)
        save_best_version(args.env, trained_agent)
    else:
        for env_name in environments:
            history, metrics, best_score, trained_agent = run_ppo(
                env_name, args.n_games
            )
            utils.plot_running_avg(history, env_name)
            df = pd.DataFrame(metrics)
            df.to_csv(f"metrics/{env_name}_metrics.csv", index=False)
            save_best_version(env_name, trained_agent)
