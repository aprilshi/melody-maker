import numpy as np

# --- Musical Constraints ---
TIME_STEPS = 32
PITCH_MAP = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 0] 
REST = 0
TEMPO = 60
TICK_PER_STEP = 120

# --- Evolution Parameters ---
POPULATION_SIZE = 6
WEIGHT_SCALE = 0.5
WEIGHT_PERTURB_STRENGTH = 0.1  # Lowered from 0.2 to keep evolution stable
P_MUTATE_WEIGHT = 0.3
P_ADD_NODE = 0.05
P_ADD_CONN = 0.1

# --- Activation Functions ---
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
def tanh(x): return np.tanh(x)
def sin(x): return np.sin(x)
def gaussian(x): return np.exp(-(x**2))

ACTIVATION_CHOICES = [sigmoid, tanh, sin, gaussian]