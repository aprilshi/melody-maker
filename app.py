import streamlit as st
import os
import random
import numpy as np
import subprocess
import sys

# Import existing modules
import config
from cppn import CPPN
from genome import Genome, crossover_genomes, mutate_genome
from melody_generation import generate_melody
from play_rate_music import save_midi_file

import matplotlib.pyplot as plt

def display_piano_roll(melody_array):
    """Generates a piano roll plot for the Streamlit UI."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Filter out rests (0) for the plot
    times = [t for t, pitch in enumerate(melody_array) if pitch != 0]
    pitches = [pitch for pitch in melody_array if pitch != 0]
    
    ax.scatter(times, pitches, marker='s', s=100, color='#1f77b4')
    
    ax.set_ylim(min(config.PITCH_MAP[0:-1]) - 1, max(config.PITCH_MAP[0:-1]) + 1)
    ax.set_xlim(-0.5, config.TIME_STEPS - 0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title("Melody Visualization")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    return fig

SOUNDFONT_FILENAME = "TimGM6mb.sf2"

# TODO: Add instrument selection?

# --- Setup Page ---
st.set_page_config(page_title="Melody Breeder", layout="wide")
st.title("Melody Breeder")

# --- Helper Functions ---

def convert_midi_to_wav_custom(midi_path, wav_path, soundfont_path):
    command = [
        'fluidsynth',
        '-ni',
        '-g', '1.0',           # Set gain to 1.0
        '-F', wav_path,        # Output file
        soundfont_path,        # Input SoundFont
        midi_path              # Input MIDI
    ]
    
    # suppressing standard output so it doesn't clutter the terminal
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_audio_file(genome_id, melody_array):
    current_dir = os.getcwd()
    midi_filename = os.path.join(current_dir, f"gen_midi_{genome_id}.mid")
    wav_filename = os.path.join(current_dir, f"gen_audio_{genome_id}.wav")
    
    # 1. Create MIDI
    try:
        save_midi_file(melody_array, midi_filename)
    except Exception as e:
        st.error(f"Error saving MIDI: {e}")
        return None, None

    # 2. Convert to WAV
    soundfont_abs = os.path.abspath(SOUNDFONT_FILENAME)
    
    if not os.path.exists(soundfont_abs):
        st.error(f"🚨 SoundFont not found at: `{soundfont_abs}`. Please check your file.")
        return None, midi_filename

    # Only generate if it doesn't exist (caching)
    if not os.path.exists(wav_filename):
        try:
            convert_midi_to_wav_custom(midi_filename, wav_filename, soundfont_abs)
        except subprocess.CalledProcessError:
            st.error("FluidSynth failed to convert audio. Check if 'fluidsynth' is installed and in your PATH.")
            return None, midi_filename
        except Exception as e:
            st.error(f"Audio conversion error: {e}")
            return None, midi_filename
    
    return wav_filename, midi_filename

def cleanup_files(file_list):
    """Removes temporary files."""
    for f in file_list:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass 

# --- Session State Initialization ---
if 'population' not in st.session_state:
    st.session_state.population = [Genome() for _ in range(config.POPULATION_SIZE)]
    st.session_state.generation = 1
    st.session_state.temp_files = []

# --- Main UI Layout ---

st.header(f"Generation: {st.session_state.generation}")
if st.button("Reset / Randomize Population"):
    st.session_state.population = [Genome() for _ in range(config.POPULATION_SIZE)]
    st.session_state.generation = 1
    st.experimental_rerun()
st.write("Listen to the melodies and check the boxes of the ones you want to keep as parents.")

with st.form("selection_form"):
    
    cols = st.columns(2) 
    selected_indices = []
    
    for i, genome in enumerate(st.session_state.population):
        col = cols[i % 2]
        
        with col:
            st.subheader(f"Melody #{i+1}")
            
            cppn = CPPN(genome)
            melody = generate_melody(cppn)
            # print(f"generation {st.session_state.generation} melody #{i+1}: {melody}")
            
            wav_path, midi_path = ensure_audio_file(f"g{st.session_state.generation}_p{i}", melody)
            
            if wav_path and os.path.exists(wav_path):
                st.session_state.temp_files.append(wav_path)
                if midi_path: st.session_state.temp_files.append(midi_path)
                
                # Display Audio Player (This is your Play button!)
                st.audio(wav_path, format='audio/wav')
            else:
                st.warning("Audio unavailable (Conversion Failed)")
            
            if st.checkbox(f"Keep Melody #{i+1}", key=f"select_{st.session_state.generation}_{i}"):
                selected_indices.append(i)

            # 2. Display Piano Roll (Visual Display)
            fig = display_piano_roll(melody)
            st.pyplot(fig)
            st.markdown("---")
            

    submitted = st.form_submit_button("Evolve Next Generation")

    if submitted:
        if len(selected_indices) == 0:
            st.error("You must select at least one parent to evolve!")
        else:
            with st.spinner("Evolving..."):
                parents = [st.session_state.population[i] for i in selected_indices]
                next_population = []
                
                # Elitism
                for p in parents:
                    next_population.append(p.clone())
                
                # Breeding
                while len(next_population) < config.POPULATION_SIZE:
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    child = crossover_genomes(parent1, parent2)
                    child = mutate_genome(child)
                    next_population.append(child)
                cleanup_files(st.session_state.temp_files)
                st.session_state.temp_files = []

                st.session_state.population = next_population
                st.session_state.generation += 1
                st.experimental_rerun()
