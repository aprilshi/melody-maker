import numpy as np
from enum import Enum
# Configuration Constants

# musical constraints
TIME_STEPS = 32           # Number of notes in the sequence (length of melody)
PITCH_LOW = 36            # C2 MIDI note
PITCH_HIGH = 88           # C8 MIDI note
REST = 0                  # MIDI note value for a rest
PITCH_RANGE = PITCH_HIGH - PITCH_LOW + 1 # 53 notes
TOTAL_OUTPUTS = PITCH_RANGE + 1 # 54 (53 pitches + 1 rest)
PITCH_MAP = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84] + [REST]
# MIDO playing Parameters
TICK_PER_STEP = 120    # MIDI ticks per time step (e.g., a 16th note) (i think this is tempo? check back)
TEMPO = 60             # BPM

# cppn topology
NUM_HIDDEN_NODES = 8

# GA Parameters
POPULATION_SIZE = 6
NUM_GENERATIONS = 10
NUM_PARENTS_TO_KEEP = 10  # Elitism: how many top genomes survive untouched
WEIGHT_SCALE = 0.5          # Scale for initial randomization of weights and biases
P_MUTATE_WEIGHT = 0.3     # Probability of a weight being mutated
P_MUTATE_ACTIVATION = 0.2 # Probability of a node's activation function changing
WEIGHT_PERTURB_STRENGTH = 0.8 # Std dev of small weight adjustments

# activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def sin(x): return np.sin(x)
def gaussian(x): return np.exp(-x**2)

class Activation(Enum):
    SIGMOID = sigmoid
    TANH = tanh
    SIN = sin
    GAUSSIAN = gaussian
ACTIVATION_CHOICES = [Activation.SIGMOID, Activation.TANH,
                      Activation.SIN, Activation.GAUSSIAN]