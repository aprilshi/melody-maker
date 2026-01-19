import streamlit as st
import os
import random
import numpy as np
import subprocess
import sys
try:
    # Newer Streamlit versions (1.12.0+)
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    try:
        # Older Streamlit versions
        from streamlit.scriptrunner import get_script_run_ctx
    except ImportError:
        # Very old Streamlit versions
        from streamlit.report_thread import get_report_ctx as get_script_run_ctx

# Import existing modules
import config
from cppn import CPPN
from genome import Genome, crossover_genomes, mutate_genome
from melody_generation import generate_melody
from play_rate_music import save_midi_file

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDFONT_FILENAME = os.path.join(BASE_DIR, "TimGM6mb.sf2")

def get_session_dir():
    """Creates a unique directory for each user session."""
    ctx = get_script_run_ctx()
    # Fallback to 'default' if the context cannot be found (e.g., during local testing)
    session_id = ctx.session_id if (ctx and hasattr(ctx, 'session_id')) else "default"
    
    session_dir = os.path.join(BASE_DIR, "temp_audio", session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def cleanup_session_files():
    """Deletes all temporary files for the current session."""
    session_dir = get_session_dir()
    for f in os.listdir(session_dir):
        try:
            os.remove(os.path.join(session_dir, f))
        except Exception:
            pass

def display_piano_roll(melody_array):
    """Renders a piano roll with fixed pitch bounds and invisible rests."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Filter out the REST value (0) to determine the vertical bounds
    active_pitches = [p for p in config.PITCH_MAP if p != 0]
    min_p, max_p = min(active_pitches), max(active_pitches)
    
    times = np.arange(len(melody_array))
    # A mask that is True only for actual MIDI notes, not rests
    mask = (melody_array != 0)
    
    # Plot only the active notes
    # Using 's' for square markers mimics a traditional piano roll
    ax.scatter(times[mask], melody_array[mask], marker='s', s=80, color='#1f77b4')
    
    # Constrain the vertical axis to exactly the range of PITCH_MAP
    ax.set_ylim(min_p - 1, max_p + 1)
    ax.set_xlim(-0.5, config.TIME_STEPS - 0.5)
    
    # Formatting for clarity
    ax.set_yticks(active_pitches)
    ax.set_ylabel("MIDI Pitch")
    ax.set_xlabel("Time Step")
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

SOUNDFONT_FILENAME = "TimGM6mb.sf2"

# TODO: Add instrument selection?

# --- Setup Page ---
st.set_page_config(page_title="Melody Breeder", layout="wide")
st.title("Melody Breeder")

# --- Helper Functions ---
def convert_midi_to_wav(midi_path, wav_path):
    """Uses FluidSynth to convert MIDI to WAV."""
    command = [
        'fluidsynth', '-ni', '-g', '1.0', '-F', wav_path, 
        SOUNDFONT_FILENAME, midi_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_audio_file(genome_id, melody_array):
    """Generates MIDI and WAV files in the session directory."""
    session_dir = get_session_dir()
    midi_filename = os.path.join(session_dir, f"gen_{genome_id}.mid")
    wav_filename = os.path.join(session_dir, f"gen_{genome_id}.wav")
    
    if not os.path.exists(SOUNDFONT_FILENAME):
        st.error(f"🚨 SoundFont not found: `{SOUNDFONT_FILENAME}`")
        return None, None

    save_midi_file(melody_array, midi_filename)
    if not os.path.exists(wav_filename):
        try:
            convert_midi_to_wav(midi_filename, wav_filename)
        except Exception as e:
            st.error(f"Conversion Error: {e}")
            return None, midi_filename
    
    return wav_filename, midi_filename

# --- Session State Initialization ---
if 'population' not in st.session_state:
    st.session_state.population = [Genome() for _ in range(config.POPULATION_SIZE)]
    st.session_state.generation = 1
    st.session_state.temp_files = []

# --- Main UI Layout ---

st.header(f"Generation: {st.session_state.generation}")
if st.button("Reset Evolution"):
    cleanup_session_files()
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

            
            # 2. Display Piano Roll (Visual Display)
            fig = display_piano_roll(melody)
            st.pyplot(fig)
            
            if st.checkbox(f"Keep Melody #{i+1}", key=f"select_{st.session_state.generation}_{i}"):
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

                cleanup_session_files() # Clear audio from old generation
                st.session_state.temp_files = []

                st.session_state.population = next_population
                st.session_state.generation += 1
                st.experimental_rerun()
