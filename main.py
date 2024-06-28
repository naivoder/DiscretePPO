import gymnasium as gym
import utils
from agent import DiscretePPOAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd
import os

os.makedirs("weights", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("environments", exist_ok=True)

warnings.simplefilter("ignore")

environments = [
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v2",
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


def run_ppo(env_name, n_games, n_epochs, horizon, batch_size, continue_training=False):
    env = gym.make(env_name, render_mode="rgb_array")
    save_prefix = env_name.split("/")[-1]

    print(f"\nEnvironment: {env_name}")
    print(f"Obs.Space: {env.observation_space.shape} Act.Space: {env.action_space.n}")
    
    preprocess = True if len(env.observation_space.shape) == 3 else False

    agent = DiscretePPOAgent(
        env_name,
        env.observation_space.shape,
        env.action_space.n,
        alpha=3e-4,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # continue training from saved checkpoint
    if continue_training:
        if os.path.exists(f"weights/{save_prefix}_actor.pt"):
            agent.load_checkpoints()

    n_steps, n_learn, best_score = 0, 0, float("-inf")
    history, metrics = [], []

    for i in range(n_games):
        state, _ = env.reset()
        if preprocess:
            state = utils.preprocess_frame(np.array(state, dtype=np.float32)).flatten()
        else:
            state = np.array(state, dtype=np.float32).flatten()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action, prob = agent.choose_action(state)

            next_state, reward, term, trunc, _ = env.step(action)
            if preprocess:
                next_state = utils.preprocess_frame(np.array(next_state, dtype=np.float32)).flatten()
            else:
                next_state = np.array(next_state, dtype=np.float32).flatten()

            agent.remember(state, next_state, action, prob, reward, term or trunc)

            n_steps += 1
            if n_steps > batch_size and n_steps % horizon == 0:
                agent.learn()
                n_learn += 1

            score += reward
            state = next_state

        history.append(score)
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        metrics.append(
            {
                "episode": i + 1,
                "average_score": avg_score,
                "best_score": best_score,
            }
        )

        print(
            f"[{env_name} Episode {i + 1:04}/{n_games}]  Average Score = {avg_score:.2f}",
            end="\r",
        )

    return history, metrics, best_score, agent


def save_results(env_name, history, metrics, agent):
    save_prefix = env_name.split("/")[-1]
    utils.plot_running_avg(history, save_prefix)
    df = pd.DataFrame(metrics)
    df.to_csv(f"metrics/{save_prefix}_metrics.csv", index=False)
    save_best_version(env_name, agent)


def save_best_version(env_name, agent, seeds=100):
    agent.load_checkpoints()

    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(seeds):
        env = gym.make(env_name, render_mode="rgb_array")
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()

        frames = []
        total_reward = 0

        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action, _ = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()
            total_reward += reward
            state = next_state

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    save_prefix = env_name.split("/")[-1]
    utils.save_animation(best_frames, f"environments/{save_prefix}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "-n",
        "--n_games",
        default=50000,
        type=int,
        help="Number of episodes (games) to run during training",
    )
    parser.add_argument(
        "--n_epochs",
        default=3,
        type=int,
        help="Number of epochs during learning",
    )
    parser.add_argument(
        "-s",
        "--n_steps",
        default=128,
        type=int,
        help="Horizon, number of steps between learning",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=256,
        type=int,
        help="Batch size for learning",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        history, metrics, best_score, trained_agent = run_ppo(
            args.env, args.n_games, args.n_epochs, args.n_steps, args.batch_size
        )
        save_results(args.env, history, metrics, trained_agent)
    else:
        for env_name in environments:
            history, metrics, best_score, trained_agent = run_ppo(
                env_name, args.n_games, args.n_epochs, args.n_steps, args.batch_size
            )
            save_results(env_name, history, metrics, trained_agent)
