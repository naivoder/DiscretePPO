import matplotlib.pyplot as plt
import numpy as np
import imageio


def collect_fixed_states(envs, n_envs, steps=5):
    """
    Collect some fixed initial states for monitoring average critic value
    This has shown to be an indicator of bugs in my code, so I'm sort of
    approximating this idea from the DQN paper (where they actually
    calculate the rollouts...) anyways, these should mostly be high value
    so we can use this value to track learning
    """
    shape = (n_envs * steps, *envs.single_observation_space.shape)
    fixed_states = np.zeros(shape, dtype=np.float32)
    for i in range(steps):
        states, _ = envs.reset()
        start = i * n_envs
        end = start + n_envs
        fixed_states[start:end, ...] = states
    return fixed_states


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


def plot_metrics(env, metrics):
    episodes = [m["episode"] for m in metrics]
    avg_scores = [m["average_score"] for m in metrics]
    avg_values = [m["average_critic_value"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Score", color="tab:blue")
    ax1.plot(episodes, avg_scores, label="Average Score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Q Value", color="tab:red")
    ax2.plot(episodes, avg_values, label="Average Critic Value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title(f"Average Score vs Average Critic Value per Episode in {env}")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_metrics.png")
    plt.close()
