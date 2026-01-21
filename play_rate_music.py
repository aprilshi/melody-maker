import mido
from mido import Message, MidiFile, MidiTrack
from midi2audio import FluidSynth
from IPython.display import Audio, display
import numpy as np
import os
import sys
import random
import config

# --- AUDIO & PATH CONFIGURATION ---
SOUNDFONT_FILENAME = "YDP-GrandPiano-20160804.sf2"  # Using specified SoundFont path

# --- MIDI Helper Functions ---

def bpm_to_tempo(bpm):
    """
    Converts Beats Per Minute (BPM) to microseconds per quarter note, 
    which is the standard MIDI tempo format.
    """
    # 60 seconds/minute * 1,000,000 microseconds/second / BPM
    return int(60 * 1000000 / bpm)


def save_midi_file(melody_array, filename, tempo = config.TEMPO):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 1. Set Tempo
    tempo_microseconds = bpm_to_tempo(tempo)
    set_tempo_message = mido.MetaMessage('set_tempo', tempo=tempo_microseconds, time=0)
    track.append(set_tempo_message)

    # 2. Set Instrument (Program Change)
    track.append(Message('program_change', program=0, time=0)) 
    
    if len(melody_array) == 0:
        mid.save(filename)
        return

    # Initialize state with the first note
    current_note = int(melody_array[0])
    current_duration = config.TICK_PER_STEP

    # Start the first note
    if current_note != config.REST:
        track.append(Message('note_on', note=current_note, velocity=80, time=0))

    # Loop starting from the second note to the end
    for i in range(1, len(melody_array)):
        next_note = int(melody_array[i])

        if next_note == current_note:
            current_duration += config.TICK_PER_STEP
        else:
            if current_note != config.REST:
                track.append(Message('note_off', note=current_note, velocity=80, time=current_duration))
            else:
                track.append(Message('note_off', note=0, velocity=0, time=current_duration))

            if next_note != config.REST:
                track.append(Message('note_on', note=next_note, velocity=80, time=0))
            
            current_note = next_note
            current_duration = config.TICK_PER_STEP

    if current_note != config.REST:
        track.append(Message('note_off', note=current_note, velocity=80, time=current_duration))
    else:
        track.append(Message('note_off', note=0, velocity=0, time=current_duration))

    try:
        mid.save(filename)
    except Exception as e:
        print(f"Error saving MIDI file: {e}")



def play_melody(midi_filename, wav_filename):
    """
    Uses FluidSynth to convert MIDI to WAV and plays the audio via IPython.display.
    Uses absolute path for the SoundFont
    """
    
    try:
        current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        soundfont_path_abs = os.path.join(current_dir, SOUNDFONT_FILENAME)
    except IndexError:
        soundfont_path_abs = os.path.abspath(SOUNDFONT_FILENAME)

    if not os.path.exists(soundfont_path_abs):
        print(f"\n--- FATAL ERROR: SoundFont File Not Found ---")
        print(f"FluidSynth cannot find the SoundFont at: {soundfont_path_abs}")
        print(f"Ensure {SOUNDFONT_FILENAME} is in the current working directory.")
        return # Skip playback and score
        
    # 1. Convert MIDI to WAV
    try:
        fs = FluidSynth(sound_font=soundfont_path_abs) 
        fs.midi_to_audio(midi_filename, wav_filename)

        # 2. Display audio player
        print("Playing audio...")
        display(Audio(wav_filename)) 

    except FileNotFoundError:
        print("\n--- Audio Playback Error (FluidSynth) ---")
        print("The 'fluidsynth' executable was not found on your system PATH.")
        print("Please ensure FluidSynth is installed and accessible.")
        print("----------------------------\n")
        
    except Exception as e:
        print(f"An unexpected error occurred during audio playback: {e}")


def get_human_score(genome_id, melody_array):
    """
    Saves the melody as a MIDI file, plays it, and prompts for human-in-the-loop scoring.
    """
    midi_filename = f"temp_g{genome_id}.mid"
    wav_filename = f"temp_g{genome_id}.wav"
    
    save_midi_file(melody_array, midi_filename) 

    print("\n" + "="*50)
    print(f"🎵 Now evaluating Genome {genome_id}...")
    print(f"Melody Notes: {melody_array[:10]}... ({len(melody_array)} total)")
    
    play_melody(midi_filename, wav_filename) 
    
    score = -1
    while not 1 <= score <= 10:
        try:
            score = int(input("\nScore (1-10) for this melody: "))
        except ValueError:
            print("Invalid input. Please enter an integer between 1 and 10.")
            score = -1
    
    # Clean up temporary files
    if os.path.exists(midi_filename):
        os.remove(midi_filename)
    if os.path.exists(wav_filename):
        os.remove(wav_filename)

    return float(score)