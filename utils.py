import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


def preprocess_frame(state):
    grayscale = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=2)


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def plot_running_avg(scores, env):
    """
    Plot the running average of scores with a window of 100 games.

    This function calculates the running average of a list of scores and
    plots the result using matplotlib. The running average is calculated
    over a window of 100 games, providing a smooth plot of score trends over time.

    Parameters
    ----------
    scores : list or numpy.ndarray
        A list or numpy array containing the scores from consecutive games.

    Notes
    -----
    This function assumes that `scores` is a list or array of numerical values
    that represent the scores obtained in each game or episode. The running
    average is computed and plotted, which is useful for visualizing performance
    trends in tasks such as games or simulations.

    Examples
    --------
    >>> scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> plot_running_avg(scores)
    This will plot a graph showing the running average of the scores over a window of 10 games.
    """
    avg = np.zeros_like(scores)
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(avg)
    plt.title("Running Average per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_running_avg.png")
