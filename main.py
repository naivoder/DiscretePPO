import gymnasium as gym
import utils
from agent import DiscretePPOAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
from preprocess import AtariEnv
from ale_py import ALEInterface, LoggerMode
from config import environments
import torch

warnings.simplefilter("ignore")
ALEInterface.setLoggerMode(LoggerMode.Error)


def run_ppo(args):
    def make_env():
        return AtariEnv(
            args.env,
            shape=(84, 84),
            repeat=4,
            clip_rewards=False,
        ).make()

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(args.n_envs)])
    save_prefix = args.env.split("/")[-1]

    print(f"\nEnvironment: {save_prefix}")
    print(f"Obs.Space: {envs.single_observation_space.shape}")
    print(f"Act.Space: {envs.single_action_space.n}")

    agent = DiscretePPOAgent(
        args.env,
        envs.single_observation_space.shape,
        envs.single_action_space.n,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    fixed_states = utils.collect_fixed_states(envs, args.n_envs)
    fixed_states = torch.tensor(fixed_states).to(agent.network.device)

    if args.continue_training:
        if os.path.exists(f"weights/{save_prefix}_actor.pt"):
            agent.load_checkpoints()

    best_score = min(envs.reward_range)
    scores = np.zeros(args.n_envs)

    history, metrics = [], []
    n_steps = 0

    states, _ = envs.reset()
    while len(history) < args.n_games:
        apvs = [agent.choose_action(state) for state in states]
        actions, probs, values = list(map(list, zip(*apvs)))

        next_states, rewards, term, trunc, _ = envs.step(actions)

        for j in range(args.n_envs):
            agent.remember(
                states[j],
                values[j],
                actions[j],
                probs[j],
                rewards[j],
                term[j] or trunc[j],
            )

            scores[j] += rewards[j]
            if term[j] or trunc[j]:
                history.append(scores[j])
                scores[j] = 0

        n_steps += 1
        if n_steps > args.batch_size and n_steps % args.horizon == 0:
            agent.learn()

        states = next_states
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        with torch.no_grad():
            _, avg_val = agent.network(fixed_states)
        avg_val = avg_val.mean().cpu().numpy()

        metrics.append(
            {
                "average_score": avg_score,
                "average_critic_value": avg_val,
            }
        )

        ep_str = f"[Ep. {n_steps:08}]"
        g_str = f"  Games = {len(history):05}/{args.n_games}"
        avg_str = f"  Avg. Score = {avg_score:.2f}"
        crit_str = f"  Avg. Value = {avg_val:.4e}"
        print(ep_str + g_str + avg_str + crit_str, end="\r")

    torch.save(agent.network.state_dict(), f"weights/{save_prefix}_final.pt")
    
    save_best_version(args.env, agent)
    utils.save_results(args.env, metrics, agent)


def save_best_version(env_name, agent, seeds=100):
    agent.load_checkpoints()

    save_prefix = env_name.split("/")[-1]
    env = AtariEnv(
        env_name,
        shape=(84, 84),
        repeat=4,
        clip_rewards=False,
    ).make()

    best_score = min(env.reward_range)
    best_frames = None

    for s in range(seeds):
        state, _ = env.reset(seed=s)

        frames = []
        total_reward = 0

        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())

            action, _, _ = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            total_reward += reward
            state = next_state

        if total_reward > best_score:
            best_score = total_reward
            best_frames = frames

    save_prefix = env_name.split("/")[-1]
    utils.save_animation(best_frames, f"environments/{save_prefix}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "--n_envs",
        default=8,
        type=int,
        help="Number of parallel environments during training",
    )
    parser.add_argument(
        "--n_games",
        default=20000,
        type=int,
        help="Total number of games to play during training",
    )
    parser.add_argument(
        "--n_epochs",
        default=10,
        type=int,
        help="Number of epochs during learning",
    )
    parser.add_argument(
        "--horizon",
        default=128,
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
        default=True,
        type=bool,
        help="Continue training from saved weights.",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights", "csv"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        run_ppo(args)
    else:
        for env_name in environments:
            args.env = env_name
            run_ppo(args)
