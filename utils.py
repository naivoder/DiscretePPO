import matplotlib.pyplot as plt
import numpy as np
import imageio

def collect_fixed_states(envs, n_envs, max_steps=50): 
    states, _ = envs.reset()
    steps = np.random.randint(1, max_steps)
    for _ in range(steps):
        actions = [envs.single_action_space.sample() for _ in range(n_envs)]
        states, _, _, _, _ = envs.step(actions)
    return states

def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def plot_running_avg(scores, env):
    avg = np.zeros_like(scores)
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(avg)
    plt.title("Running Average per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"metrics/{env}_running_avg.png")
    plt.close()


def plot_critic_val(vals, env):
    plt.plot(vals)
    plt.title("Average Critic Value of Fixed States")
    plt.xlabel("Episode")
    plt.ylabel("Average Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"metrics/{env}_critic_val.png")
    plt.close()