import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd

def save_results(env_name, metrics, agent):
    save_prefix = env_name.split("/")[-1]
    df = pd.DataFrame(metrics)
    df.to_csv(f"csv/{save_prefix}_metrics.csv", index=False)
    plot_metrics(save_prefix, df)
    

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
    # episodes = np.array(metrics["episode"])
    run_avg_scores = np.array(metrics["average_score"])
    avg_values = np.array(metrics["average_critic_value"])
    episodes = np.arange(len(run_avg_scores))

    run_avg_vals = np.zeros_like(avg_values)
    for i in range(len(avg_values)):
        run_avg_vals[i] = np.mean(avg_values[max(0, i - 100) : i + 1])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Score", color="tab:blue")
    ax1.plot(episodes, run_avg_scores, label="Average Score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Critic Value", color="tab:red")
    ax2.plot(episodes, run_avg_vals, label="Average Critic Value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title(f"Average Score vs Average Critic Value per Episode in {env}")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_metrics.png")
    plt.close()
