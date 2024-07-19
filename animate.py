from preprocess import AtariEnv
import torch
from agent import DiscretePPOAgent
from argparse import ArgumentParser
from utils import save_animation

def generate_animation(env_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AtariEnv(
            env_name,
            shape=(84, 84),
            repeat=4,
            clip_rewards=True,
            no_ops=0,
            fire_first=False,
        ).make()
    agent = DiscretePPOAgent(
        env_name,
        env.observation_space.shape,
        env.action_space.n)
    
    agent.load_checkpoints()
    
    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(10):
        frames = []
        total_reward = 0

        state, _ = env.reset()
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
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
