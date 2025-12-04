import numpy as np
from cppn import CPPN
import config
# --- Melody Generation Function ---
def generate_melody(cppn: CPPN, num_timesteps = config.TIME_STEPS):
    """
    Generates a sequence of MIDI notes (or REST=0) from the CPPN.

    The output node index maps directly to:
    [0] -> MIDI note PITCH_LOW (36)
    ...
    [PITCH_RANGE-1] -> MIDI note PITCH_HIGH (88)
    [PITCH_RANGE] -> REST (0)
    """
    melody_array = []

    pitch_map = config.PITCH_MAP

    # # Map the timestep [0, num_timesteps-1] to a normalized input range [-1, 1]
    # time_points = np.linspace(-1, 1, num_timesteps)

    for t in range(num_timesteps):
        # get preference vector from CPPN
        pitch_preferences = cppn.forward_pass(t)

        # apply Softmax to convert raw preferences into a probability distribution
        # add a small epsilon (1e-6) for numerical stability
        exp_preferences = np.exp(pitch_preferences - np.max(pitch_preferences))
        probabilities = exp_preferences / (np.sum(exp_preferences) + 1e-6)

        # pick the index with the highest probability
        # This index directly corresponds to an entry in pitch_map
        selected_index = np.argmax(probabilities)

        # Map the selected index to the actual MIDI note or REST value
        selected_note = pitch_map[selected_index]

        melody_array.append(selected_note)

    return np.array(melody_array)

