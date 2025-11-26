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


def save_midi_file(melody_array, filename):
    """
    Converts a sequence of MIDI notes into a playable MIDI file, 
    using constants imported from the 'config' module.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 1. Set Tempo
    tempo_microseconds = bpm_to_tempo(config.TEMPO)
    set_tempo_message = mido.MetaMessage('set_tempo', tempo=tempo_microseconds, time=0)
    track.append(set_tempo_message)

    # 2. Set Instrument (Program Change)
    track.append(Message('program_change', program=1, time=0)) # Using Piano

    for note in melody_array:
        note = int(note) 
        if note != config.REST:
            # Note On: velocity 64 (your preference)
            track.append(Message('note_on', note=note, velocity=64, time=0))
            # Note Off: duration is TICK_PER_STEP
            track.append(Message('note_off', note=note, velocity=64, time=config.TICK_PER_STEP))
        else:
            # Rest: time delay is TICK_PER_STEP. Use a silent note_off for the duration.
            track.append(Message('note_off', note=0, velocity=0, time=config.TICK_PER_STEP))

    try:
        mid.save(filename)
    except Exception as e:
        print(f"Error saving MIDI file: {e}")


def play_melody(midi_filename, wav_filename):
    """
    Uses FluidSynth to convert MIDI to WAV and plays the audio via IPython.display.
    Uses absolute path for the SoundFont
    """
    
    # We find the directory where the calling script is and combine it with the filename.
    try:
        current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        soundfont_path_abs = os.path.join(current_dir, SOUNDFONT_FILENAME)
    except IndexError:
        # Fallback for when running directly in IPython without sys.argv[0] defined
        soundfont_path_abs = os.path.abspath(SOUNDFONT_FILENAME)

    if not os.path.exists(soundfont_path_abs):
        print(f"\n--- FATAL ERROR: SoundFont File Not Found ---")
        print(f"FluidSynth cannot find the SoundFont at: {soundfont_path_abs}")
        print("Ensure 'YDP-GrandPiano-20160804.sf2' is in the current working directory.")
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