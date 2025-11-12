# melody-maker
Picbreeder but for simple melodies. Options for direct encoding and using HyperNEAT to evolve CPPN

## Usage
### 1. Create the environment ('venv')
python -m venv venv

### 2. Activate the environment (Command depends on your OS)
#### For macOS/Linux:
source venv/bin/activate
#### For Windows (Command Prompt):
venv\Scripts\activate

### 3. Install Dependencies**

This installs all the packages listed in your `requirements.txt`.

```bash
pip install -r requirements.txt
```

\*\*CRITICAL NOTE on FluidSynth:** The `midi2audio` package requires the external **FluidSynth** executable to be installed on your operating system (it is not a Python package). If you see a `FileNotFoundError` during playback, you will need to install it manually.

For macOS with homebrew:
```
brew install fluidsynth
```

### 4. Run the Python Script**

Execute your main file:

```bash
python main.py
```
### What to Expect When Running

The `run_evolution` loop will begin. For every genome:

1.  It will calculate the melody array.
2.  It will save the melody to a temporary MIDI file.
3.  It will convert that MIDI file to a temporary WAV file using **FluidSynth** and display an audio player (if running in a Jupyter-like environment, otherwise it will just print the score prompt).
4.  You will be prompted to enter a score from 1 to 10 based on how the melody sounds.

This score is then used as the fitness value to breed the next generation of musical CPPNs!