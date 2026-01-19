import numpy as np
from cppn import CPPN
import config
# --- Melody Generation Function ---
def generate_melody(cppn: CPPN, num_timesteps = config.TIME_STEPS):
    melody_array = []
    pitch_map = config.PITCH_MAP

    for t in range(num_timesteps):
        normalized_t = (2.0 * t / (num_timesteps - 1)) - 1.0
        
        # Pass the normalized time to the CPPN
        pitch_preferences = cppn.forward_pass(normalized_t)

        exp_preferences = np.exp(pitch_preferences - np.max(pitch_preferences))
        probabilities = exp_preferences / (np.sum(exp_preferences) + 1e-6)
        selected_index = np.argmax(probabilities)
        melody_array.append(pitch_map[selected_index])

    return np.array(melody_array)