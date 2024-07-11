import gymnasium as gym
import utils
from agent import DiscretePPOAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd
from preprocess import AtariEnv

warnings.simplefilter("ignore")

environments = [
    # "CartPole-v1",  # gymnasium environments
    # "MountainCar-v0",
    # "Acrobot-v1",
    # "LunarLander-v2",
    "AsteroidsNoFrameskip-v4",  # atari environments
    "BreakoutNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "DonkeyKongNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "FroggerNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MarioBrosNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "TetrisNoFrameskip-v4",
    "VideoChessNoFrameskip-v4",
]


def run_ppo(args):
    if "ALE" in args.env or "NoFrameskip" in args.env:
        env = AtariEnv(
            args.env,
            shape=(84, 84),
            repeat=4,
            clip_rewards=True,
        ).make()
    else:
        env = gym.make(args.env, render_mode="rgb_array")
    save_prefix = args.env.split("/")[-1]

    print(f"\nEnvironment: {save_prefix}")
    print(f"Obs.Space: {env.observation_space.shape} Act.Space: {env.action_space.n}")

    agent = DiscretePPOAgent(
        args.env,
        env.observation_space.shape,
        env.action_space.n,
        alpha=1e-4,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    if args.continue_training:
        if os.path.exists(f"weights/{save_prefix}_actor.pt"):
            agent.load_checkpoints()

    n_steps, n_learn, best_score = 0, 0, float("-inf")
    history, metrics = [], []

    for i in range(args.n_games):
        state, _ = env.reset()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action, prob = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            agent.remember(state, next_state, action, prob, reward, term or trunc)

            n_steps += 1
            if n_steps > args.batch_size and n_steps % args.horizon == 0:
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
            f"[{save_prefix} Episode {i + 1:04}/{args.n_games}]  Average Score = {avg_score:.2f}",
            end="\r",
        )

    save_results(args.env, history, metrics, agent)


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
        # you know, I probably could just reset the environment...
        # is reinitializing helping anything?
        if "ALE" in env_name or "NoFrameskip" in args.env:
            env = AtariEnv(
                env_name,
                shape=(84, 84),
                repeat=4,
                clip_rewards=True,
                no_ops=0,
                fire_first=False,
            ).make()
        else:
            env = gym.make(env_name, render_mode="rgb_array")

        save_prefix = env_name.split("/")[-1]
        state, _ = env.reset()

        frames = []
        total_reward = 0

        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action, _ = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

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
        "--n_games",
        default=10000,
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
        "--horizon",
        default=1024,
        type=int,
        help="Horizon, number of steps between learning",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size for learning",
    )
    parser.add_argument(
        "--continue_training",
        default=False,
        type=bool,
        help="Continue training from saved weights.",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        run_ppo(args)
    else:
        for env_name in environments:
            args.env = env_name
            run_ppo(args)
