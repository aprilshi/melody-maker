# melody-maker
Picbreeder but for simple melodies. Options for direct encoding and using HyperNEAT to evolve CPPN

## Usage
### 1. Create the environment ('venv')
```
python -m venv venv
```
### 2. Activate the environment (Command depends on your OS)
#### For macOS/Linux:
```
source venv/bin/activate
```
#### For Windows (Command Prompt):
```
venv\Scripts\activate
```
### 3. Install Dependencies and soundfont file**

```bash
pip install -r requirements.txt
```

\*\*CRITICAL NOTE on FluidSynth:** The `midi2audio` package requires the external **FluidSynth** executable to be installed (it is not a Python package). If you see a `FileNotFoundError` during playback, you will need to install it manually.

For macOS with homebrew:
```
brew install fluidsynth
```

Now go online and find a soundfont file (.sf2), download it locally and set SOUNDFONT_FILENAME = "local_path"

### 4. Run the Python Script

#### A. Streamlit UI
Open Streamlit app:

```
streamlit run app.py
```

#### What to Expect When Running
1.  A new tab will open in your web browser
2. You will see Generation 1 of melodies.
3. Play melodies and check the boxes next to melodies that sound interesting or not terrible.
4. Click "Evolve Next Generation"
5. The page will refresh with Generation 2, descendants of your choices.


#### B. Command Line**
\*\*Note that the command line run is not as actively maintained as the streamlit.

Execute your main file:

```bash
ipython

>> %run main.py
```
#### What to Expect When Running

The `run_evolution` loop will begin. For every genome:

1.  It will calculate the melody array.
2.  It will save the melody to a temporary MIDI file.
3.  It will convert that MIDI file to a temporary WAV file using **FluidSynth** and play the generated melody.
4.  You will be prompted to enter a score from 1 to 10 based on how the melody sounds.

This score is then used as the fitness value to breed the next generation of musical CPPNs!

### 5. Make adjustments!

The majority of parameters are defined in `config.py`, feel free to mess with things like population size or note range.