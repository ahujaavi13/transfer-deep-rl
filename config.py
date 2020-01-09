"""
Configuration Variables and hyper-parameters
"""

# Transfer Methods
TRANSFER_METHOD = [0, 1, 2, 3, 4, 5, 6]
BASELINE = TRANSFER_METHOD[0]  # No Transfer

#  Plot labels (descriptions) for methods in TRANSFER_METHOD
PLOT_LABELS = [
    'Without TL', '3rd Layer weights to 1st Layer',
    '2nd Layer weights to 2nd Layer', '2nd Layer weights to 1st Layer',
    'Unfrozen matrix concatenation - bottom half zero', 'matrix concatenation over state space',
    'Replicated Rows unfrozen',
]

# Training Specifics
EPISODE_COUNT = 5000  # Episode count
TICKS = 1000  # Max ticks per episode
EPS_START = 1.0  # Initial value of Epsilon (Exploration Factor)
EPS_END = 0.01  # Min Value of Epsilon
EPS_DECAY = 0.995  # Decay rate

OUT_WEIGHTS_FILENAME = 'weights/learned_weights.pth'  # Filename for saving weights
PLOT_SCORES_FREQUENCY = 100  # Frequency of scores used to plot graph and print output to console.

# Environment Specifics
ENVIRONMENTS = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0']  # List of OpenAI Gym Environment
PRE_TRAINED_ENV = ENVIRONMENTS[2]  # Pre-trained environment.
SELECTED_ENV = ENVIRONMENTS[0]  # Current environment to experiment

# Agent specifics
REPLAY_BUFFER_SIZE = int(1e5)  # Replay buffer
BATCH_SIZE = 64  # Mini-batch
GAMMA = 1.0  # Diminishing rewards if less that 1
TAU = 1e-3  # Target weights update
LEARNING_RATE = 5e-4  # learning rate
UPDATE_FREQUENCY = 4  # Learned parameters update frequency

# Parameters per layer
FC1_UNITS = 192
FC2_UNITS = 192

# Pre-trained weights directory
BASE_DIRECTORY = 'weights'
FILENAME = PRE_TRAINED_ENV
