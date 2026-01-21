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
import networkx as nx
import matplotlib.patches as mpatches

def draw_neural_network(genome):
    """Generates a graph visualization with a color-coded legend and activation labels."""
    G = nx.DiGraph()
    
    # Define colors
    color_input = '#4caf50'  # Green
    color_hidden = '#2196f3' # Blue
    color_output = '#f44336' # Red

    node_colors = []
    node_labels = {}

    for node_id, node in genome.nodes.items():
        G.add_node(node_id)
        # Add labels to show activation functions (e.g., 'sin', 'tanh')
        # Using .__name__ gets the function name for display
        label = f"{node_id}\n({node.activation.__name__})" if node.type == 'hidden' else str(node_id)
        node_labels[node_id] = label
        
        if node.type == 'input':
            node_colors.append(color_input)
            G.nodes[node_id]['layer'] = 0
        elif node.type == 'output':
            node_colors.append(color_output)
            G.nodes[node_id]['layer'] = 2
        else:
            node_colors.append(color_hidden)
            G.nodes[node_id]['layer'] = 1

    for (in_id, out_id), conn in genome.connections.items():
        if conn.enabled:
            # Scale edge thickness by weight
            G.add_edge(in_id, out_id, weight=abs(conn.weight) * 2)

    pos = nx.multipartite_layout(G, subset_key="layer")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw the edges first
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color='gray', alpha=0.5)
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8)

    # Create the legend
    legend_patches = [
        mpatches.Patch(color=color_input, label='Input (Time/Bias)'),
        mpatches.Patch(color=color_hidden, label='Hidden (Processing)'),
        mpatches.Patch(color=color_output, label='Output (Pitches)')
    ]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_title(f"Gen {st.session_state.generation} - Neural Topology")
    plt.tight_layout()
    return fig

# --- Setup Paths & Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

def ensure_audio_file(genome_id, melody_array, tempo = config.TEMPO):
    """Generates unique MIDI/WAV files in session directory."""
    session_dir = get_session_dir()
    midi_filename = os.path.join(session_dir, f"gen_{genome_id}.mid")
    wav_filename = os.path.join(session_dir, f"gen_{genome_id}.wav")
    
    if not os.path.exists(SOUNDFONT_FILENAME):
        st.error(f"🚨 SoundFont not found: `{SOUNDFONT_FILENAME}`")
        return None, None

    save_midi_file(melody_array, midi_filename, tempo=tempo)
    if not os.path.exists(wav_filename):
        try:
            convert_midi_to_wav(midi_filename, wav_filename)
        except Exception as e:
            st.error(f"Conversion Error: {e}")
            return None, midi_filename
    
    return wav_filename, midi_filename

# --- UI Initialization ---


st.set_page_config(page_title="Melody Breeder", layout="wide")
st.title(":musical_keyboard: Melody Breeder")

st.sidebar.header("Evolution Settings")
st.sidebar.info("Will be applied to the next evolution.")

# 1. Structural Mutation Probabilities
p_add_node = st.sidebar.slider("Prob: Add Node", 0.0, 0.5, 0.05, help="Chance to add a new hidden neuron.")
p_add_conn = st.sidebar.slider("Prob: Add Connection", 0.0, 0.5, 0.10, help="Chance to bridge two existing neurons.")

# 2. Weight Mutation Settings
p_mutate_weight = st.sidebar.slider("Prob: Mutate Weight", 0.0, 1.0, 0.30)
weight_perturb = st.sidebar.slider("Weight Perturbation Strength", 0.01, 0.5, 0.05)

st.sidebar.markdown("---")

# Store these in a dictionary to pass around easily
current_config = {
    "P_ADD_NODE": p_add_node,
    "P_ADD_CONN": p_add_conn,
    "P_MUTATE_WEIGHT": p_mutate_weight,
    "WEIGHT_PERTURB": weight_perturb,
}

# TODO: Let user mess with configs but perhaps apply resets evolution

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
        # Inside the display loop
        melody = generate_melody(cppn)
        wav_path, midi_path = ensure_audio_file(f"g{st.session_state.generation}_p{i}", melody)
        
        if wav_path and os.path.exists(wav_path):
            st.audio(wav_path, format='audio/wav')
        
        # Plotting
        with st.expander("View Pitch Plot", expanded = True):
            st.pyplot(display_piano_roll(melody))

        # neural network map
        with st.expander("View Neural Network Diagram"):
            st.write("This diagram shows the evolved connections between the 'Time' and 'Bias' inputs and the 'Pitch' outputs.")
            net_fig = draw_neural_network(genome)
            st.pyplot(net_fig)

        # DOWNLOAD BUTTON
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
                    child = mutate_genome(
                        child, 
                        p_node=current_config["P_ADD_NODE"], 
                        p_conn=current_config["P_ADD_CONN"],
                        p_weight=current_config["P_MUTATE_WEIGHT"],
                        perturbation=current_config["WEIGHT_PERTURB"]
                    )
                    next_population.append(child)

                cleanup_session_files() # Clear disk space
                st.session_state.population = next_population
                st.session_state.generation += 1
                st.experimental_rerun()