import mido
from mido import Message, MidiFile, MidiTrack
from IPython.display import display, Audio
from midi2audio import FluidSynth

import config

SOUNDFONT_PATH = "YDP-GrandPiano-20160804.sf2"

def bpm_to_tempo(bpm):
    # Convert BPM to microseconds per beat
    return int(500000 / (bpm / 120))

def save_midi_file(melody, filename="melody.mid"):
  mid = MidiFile()
  track = MidiTrack()
  mid.tracks.append(track)
  # Set to piano
  track.append(mido.Message('program_change', program=1, time=0))
  tempo_microseconds = bpm_to_tempo(config.TEMPO)
  set_tempo_message = mido.MetaMessage('set_tempo', tempo=tempo_microseconds, time=0)
  track.append(set_tempo_message)

  for note in melody:
    if note == 0:
      track.append(Message('note_off', note=0, velocity=0, time=config.TICK_PER_STEP))
      continue
    track.append(Message('note_on', note=note, velocity=64, time=0))
    track.append(Message('note_off', note=note, velocity=64, time=config.TICK_PER_STEP))

  mid.save(filename)

def play_midi_file(filename):
    output_wav_path = "output.wav"
    # Path to soundfont
    fs = FluidSynth(sound_font = SOUNDFONT_PATH)
    fs.midi_to_audio(filename, output_wav_path)
    return Audio(output_wav_path)


def get_human_score(melody_array):
    """
    Simulates the human-in-the-loop scoring process.
    In a real application, this is where MIDI playback would occur.
    """
    print("\n" + "="*50)
    print(f"🎵 Generated Melody Array (Length {len(melody_array)}):")
    print(melody_array)
    print("="*50)

    filename = "cppn_melody_test.mid"
    save_midi_file(melody_array, filename)
    display(play_midi_file(filename))

    score = -1
    while not 1 <= score <= 10:
        try:
            score = int(input("Please listen to the melody and assign a score (1-10): "))
        except ValueError:
            print("Invalid input. Please enter an integer between 1 and 10.")
            score = -1
    return float(score)

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

    # Create the mapping array: MIDI notes + REST at the end
    # [36, 37, ..., 88, 0] where 0 is the REST
    pitch_map = list(range(config.PITCH_LOW, config.PITCH_HIGH + 1)) + [config.REST]

    # Map the timestep [0, num_timesteps-1] to a normalized input range [-1, 1]
    # This range is arbitrary but common for CPPNs
    time_points = np.linspace(-1, 1, num_timesteps)

    for t_norm in time_points:
        # Get preference vector (length 54)
        pitch_preferences = cppn.forward_pass(t_norm)

        # Apply Softmax to convert raw preferences into a probability distribution
        # The probability distribution is what allows us to "pick the highest probability"
        # We add a small epsilon (1e-6) for numerical stability
        exp_preferences = np.exp(pitch_preferences - np.max(pitch_preferences))
        probabilities = exp_preferences / (np.sum(exp_preferences) + 1e-6)

        # Selection: Pick the index with the highest probability
        # This index [0 to 53] directly corresponds to an entry in pitch_map
        selected_index = np.argmax(probabilities)

        # Map the selected index to the actual MIDI note or REST value
        selected_note = pitch_map[selected_index]

        melody_array.append(selected_note)

    return np.array(melody_array)