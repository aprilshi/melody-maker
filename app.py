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
from play_rate_music import save_midi_file, 

SOUNDFONT_FILENAME = "Sonatina_Symphonic_Orchestra.sf2"

# TODO: FIX THIS! And also maybe add an instrument selection

# --- Setup Page ---
st.set_page_config(page_title="Evolutionary Melody Maker", layout="wide")
st.title("Evolutionary Melody Maker")

# --- Helper Functions ---

def convert_midi_to_wav_custom(midi_path, wav_path, soundfont_path):
    """
    Directly calls FluidSynth with the correct flag order to prevent
    playback and ensure file generation.
    """
    # Command structure: fluidsynth -ni -F {wav} {sf2} {mid}
    # -ni: No interactive mode (don't run a shell)
    # -F: Render to file (Output)
    # -g: Gain (volume, optional)
    command = [
        'fluidsynth',
        '-ni',
        '-g', '1.0',           # Set gain to 1.0
        '-F', wav_path,        # Output file
        soundfont_path,        # Input SoundFont
        midi_path              # Input MIDI
    ]
    
    # Run the command, suppressing standard output so it doesn't clutter the terminal
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_audio_file(genome_id, melody_array):
    """
    Generates MIDI and WAV files for a specific genome.
    Returns the path to the WAV file for playback.
    """
    # Use absolute paths to avoid any directory confusion
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
            
            wav_path, midi_path = ensure_audio_file(f"g{st.session_state.generation}_p{i}", melody)
            
            if wav_path and os.path.exists(wav_path):
                st.session_state.temp_files.append(wav_path)
                if midi_path: st.session_state.temp_files.append(midi_path)
                
                # Display Audio Player (This is your Play button!)
                st.audio(wav_path, format='audio/wav')
            else:
                st.warning("Audio unavailable (Conversion Failed)")
            
            if st.checkbox(f"Keep Melody #{i+1}", key=f"select_{i}"):
                selected_indices.append(i)
            
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
                
                st.session_state.population = next_population
                st.session_state.generation += 1
                st.experimental_rerun()
