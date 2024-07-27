from preprocess import AtariEnv
import torch
from agent import DiscretePPOAgent
from argparse import ArgumentParser
from utils import save_animation

def generate_animation(env_name):
    env = AtariEnv(
            env_name,
            shape=(84, 84),
            repeat=4,
            clip_rewards=False,
            no_ops=0,
            fire_first=False,
        ).make()
    
    agent = DiscretePPOAgent(
        env_name,
        env.observation_space.shape,
        env.action_space.n)
    
    # agent.load_checkpoints()
    agent.network.load_state_dict(torch.load(f"weights/{env_name}_final.pt"))

    best_total_reward = min(env.reward_range)
    best_frames = None

    for _ in range(10):
        frames = []
        total_reward = 0

        state, _ = env.reset()
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            with torch.no_grad():
                action, _, _ = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            state = next_state
            total_reward += reward

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    save_animation(best_frames, f"environments/{env_name}.gif")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", required=True, help="Environment name from Gymnasium"
    )
    args = parser.parse_args()
    generate_animation(args.env)
