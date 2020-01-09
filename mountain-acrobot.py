import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import gym

from agent import Agent

import config as c

env = gym.make(c.SELECTED_ENV)
s_size = env.observation_space.shape[0]
a_size = env.action_space.n
env.seed(0)


def start(episodes, ticks, eps, eps_end, eps_decay, transfer_method=c.BASELINE):

    """
    Parameters:
        episodes (int): Max number of episodes for training
        ticks (int): Max number of ticks per episode
        eps (float): Starting value of exploration factor. Is reduced by decay factor
        eps_end (float): Minimum value of exploration factor
        eps_decay (float): Epsilon decay factor. Common value is 0.995
        transfer_method (int): Takes an integer value depending transfer learning experiment. Default - Baseline
    """

    print(c.PRE_TRAINED_ENV + " to " + c.SELECTED_ENV + " with Transfer Method - " + str(c.PLOT_LABELS[transfer_method]))
    episode_scores = []  # list containing scores from each episode
    scores_queue = deque(maxlen=c.PLOT_SCORES_FREQUENCY)  # Queue having last 'PLOT_SCORES_FREQUENCY' scores
    for e in range(1, episodes + 1):
        s = env.reset()
        score = 0
        for tick in range(ticks):
            env.render()  # Comment this if using Colab/Jupyter
            a = agent.act(s, transfer_method, eps)
            next_s, r, d, info = env.step(a)
            agent.step(s, a, r, next_s, d, transfer_method)
            s = next_s
            score += r
            if d:
                break
        scores_queue.append(score)  # Append recent score
        eps = max(eps_decay * eps, eps_end)  # Decay epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_queue)), end="")
        if e % c.PLOT_SCORES_FREQUENCY == 0:
            episode_scores.append(score)  # Save recent score
            print('\rEpisode {}\tMean Score: {:.2f}'.format(e, np.mean(scores_queue)))
        if e == c.EPISODE_COUNT:  # End training
            print('\nMax episodes reached. Episodes {:d} \tAverage Score: {:.2f}'.format(e - c.PLOT_SCORES_FREQUENCY,
                                                                                         np.mean(scores_queue)))
            if transfer_method == 0:  # Save scores only for Baseline
                torch.save(agent.local_dqn.state_dict(), c.OUT_WEIGHTS_FILENAME)
            break
    return episode_scores


for i in c.TRANSFER_METHOD:
    agent = Agent(s_size, a_size, seed=0)
    scores = start(c.EPISODE_COUNT, c.TICKS, c.EPS_START, c.EPS_END, c.EPS_DECAY, transfer_method=i)
    plt.plot(np.arange(100, (len(scores) + 1) * 100, 100), scores, label=(c.PLOT_LABELS[i], np.mean(scores)))

plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.suptitle('Episode V/s Reward Plot')
plt.legend()
plt.show()
env.close()
