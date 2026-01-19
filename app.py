import streamlit as st
import os
import random
import numpy as np
import subprocess
import sys
import matplotlib.pyplot as plt

# Version-agnostic context retrieval
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    try:
        from streamlit.scriptrunner import get_script_run_ctx
    except ImportError:
        from streamlit.report_thread import get_report_ctx as get_script_run_ctx

# Import existing modules
import config
from cppn import CPPN
from genome import Genome, crossover_genomes, mutate_genome
from melody_generation import generate_melody
from play_rate_music import save_midi_file

# --- Setup Paths & Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this matches your file or system path
SOUNDFONT_FILENAME = os.path.join(BASE_DIR, "TimGM6mb.sf2")

# --- Helper Functions ---

def get_session_dir():
    """Creates a unique directory for each user session to prevent overlap."""
    ctx = get_script_run_ctx()
    session_id = ctx.session_id if (ctx and hasattr(ctx, 'session_id')) else "default"
    session_dir = os.path.join(BASE_DIR, "temp_audio", session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def cleanup_session_files():
    """Deletes all temporary files for the current session."""
    session_dir = get_session_dir()
    if os.path.exists(session_dir):
        for f in os.listdir(session_dir):
            try:
                os.remove(os.path.join(session_dir, f))
            except Exception:
                pass

def display_piano_roll(melody_array):
    """Renders a piano roll with fixed pitch bounds and invisible rests."""
    fig, ax = plt.subplots(figsize=(10, 3))
    active_pitches = [p for p in config.PITCH_MAP if p != 0]
    min_p, max_p = min(active_pitches), max(active_pitches)
    
    times = np.arange(len(melody_array))
    mask = (melody_array != 0) # Hide rests
    
    ax.scatter(times[mask], melody_array[mask], marker='s', s=80, color='#1f77b4')
    ax.set_ylim(min_p - 1, max_p + 1)
    ax.set_xlim(-0.5, config.TIME_STEPS - 0.5)
    ax.set_yticks(active_pitches)
    ax.set_ylabel("MIDI Pitch")
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def convert_midi_to_wav(midi_path, wav_path):
    """Uses FluidSynth for synthesis."""
    command = [
        'fluidsynth', '-ni', '-g', '1.0', '-F', wav_path, 
        SOUNDFONT_FILENAME, midi_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_audio_file(genome_id, melody_array):
    """Generates unique MIDI/WAV files in session directory."""
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

# --- UI Initialization ---
st.set_page_config(page_title="Melody Breeder", layout="wide")
st.title("Melody Breeder")

if 'population' not in st.session_state:
    st.session_state.population = [Genome() for _ in range(config.POPULATION_SIZE)]
    st.session_state.generation = 1

st.header(f"Generation: {st.session_state.generation}")

if st.button("Reset Evolution"):
    cleanup_session_files()
    st.session_state.population = [Genome() for _ in range(config.POPULATION_SIZE)]
    st.session_state.generation = 1
    st.experimental_rerun()

# --- 1. DISPLAY SECTION (Outside form for Downloads) ---
st.write("Listen to the melodies and view their structure. If you generate interesting melodies, please download and email to me, so I can make a collection of fun community-made melodies!")
cols = st.columns(2)

for i, genome in enumerate(st.session_state.population):
    col = cols[i % 2]
    with col:
        st.subheader(f"Melody #{i+1}")
        
        # Generate melody logic
        cppn = CPPN(genome)
        melody = generate_melody(cppn)
        
        wav_path, midi_path = ensure_audio_file(f"g{st.session_state.generation}_p{i}", melody)
        
        if wav_path and os.path.exists(wav_path):
            st.audio(wav_path, format='audio/wav')
        
        # Plotting
        st.pyplot(display_piano_roll(melody))

        # DOWNLOAD BUTTONS (Now valid because they are outside st.form)
        if midi_path and os.path.exists(midi_path):
            with open(midi_path, "rb") as f:
                st.download_button(
                    label=f"Download MIDI #{i+1}",
                    data=f,
                    file_name=f"generation_{st.session_state.generation}_melody_{i+1}.mid",
                    mime="audio/midi",
                    key=f"dl_{st.session_state.generation}_{i}"
                )
        st.markdown("---")

# --- 2. SELECTION SECTION (Inside Form) ---
with st.form("selection_form"):
    st.write("### Select parents for the next generation")
    
    # We use columns to display checkboxes neatly
    check_cols = st.columns(config.POPULATION_SIZE)
    selected_indices = []
    
    for i in range(config.POPULATION_SIZE):
        with check_cols[i]:
            if st.checkbox(f"Keep #{i+1}", key=f"select_{st.session_state.generation}_{i}"):
                selected_indices.append(i)
    
    submitted = st.form_submit_button("Evolve Next Generation")

    if submitted:
        if len(selected_indices) == 0:
            st.error("Select at least one parent!")
        else:
            with st.spinner("Evolving..."):
                parents = [st.session_state.population[idx] for idx in selected_indices]
                next_population = []
                
                # Elitism
                for p in parents:
                    next_population.append(p.clone())
                
                # Breeding & Mutation
                while len(next_population) < config.POPULATION_SIZE:
                    p1, p2 = random.choice(parents), random.choice(parents)
                    child = crossover_genomes(p1, p2)
                    next_population.append(mutate_genome(child))

                cleanup_session_files() # Clear disk space
                st.session_state.population = next_population
                st.session_state.generation += 1
                st.experimental_rerun()