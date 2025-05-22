import os
import time
import soundfile as sf

from tqdm import tqdm
from pprint import pprint
from pedalboard import load_plugin 
from mido import MidiFile, Message
from pedalboard.io import AudioFile

# SERUM_PLUGIN_PATH = "/Library/Audio/Plug-Ins/Components/Serum.component" # "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
SERUM_PLUGIN_PATH = os.path.expanduser("/Users/bliu/Library/Audio/Plug-Ins/Components/Serum.component")
PLUGIN_NAME = "Serum" #"Serum 2" # "Serum 2 FX"
VSTPRESET_DIR = "../vstpreset"
MIDI_DIR = "../midi/maestro_v3"
RENDERED_AUDIO_DIR = "../rendered_audio"
SAMPLE_RATE = 44100.0  # Hz
BUFFER_SIZE = 512      # Samples per processing block (power of 2 often good)
OUTPUT_CHANNELS = 2
TAIL_DURATION_SECONDS = 3.0  # Extra time to capture release tails after MIDI ends
MIDI_DURATION_THRESHOLD = 180.0 - TAIL_DURATION_SECONDS # 3 minutes
STEM = "lead"
YEAR = "2018"

os.makedirs(RENDERED_AUDIO_DIR, exist_ok=True)


stem_mapping = {
    "lead": "LD",
    "bass": "BA",
    "keys": "KY",
    "pad": "PD",
    "pluck": "PL",
}


def get_all_vstpresets(preset_dir, stem=None):
    vstpreset_paths = []
    if stem is None:
        for stem in os.listdir(preset_dir):
            if stem != ".DS_Store":
                for p in os.listdir(os.path.join(preset_dir, stem)):
                    if p.endswith('.vstpreset'):
                        vstpreset_paths.append(os.path.join(preset_dir, stem, p))
    else:
        for p in os.listdir(os.path.join(preset_dir, stem)):
            if p.endswith('.vstpreset'):
                vstpreset_paths.append(os.path.join(preset_dir, stem, p))
    
    return vstpreset_paths



def get_all_midi_files(midi_dir, year=None):
    midi_file_paths = []
    if year is None:
        for year in os.listdir(midi_dir):
            if year != ".DS_Store":
                for midi_file in os.listdir(os.path.join(midi_dir, year)):
                    if midi_file.endswith('.midi'):
                        midi_file_paths.append(os.path.join(midi_dir, year, midi_file))
    else:
        for midi_file in os.listdir(os.path.join(midi_dir, year)):
            if midi_file.endswith('.midi'):
                midi_file_paths.append(os.path.join(midi_dir, year, midi_file))
    
    return midi_file_paths



def get_all_serum_states(vstpreset_paths, length=0):
    # Get Serum Plugin
    serum_dict = {}

    for i in tqdm(range(length), desc="Loading VST Presets"):
        serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)
        with open(vstpreset_paths[i], 'rb') as f:
            serum.raw_state = f.read()
            serum_dict[vstpreset_paths[i]] = serum

    return serum_dict


def get_midi_messages(midi_file_path):
    # TODO: Find informative parts
    midi_file = MidiFile(midi_file_path)
    midi_messages = []
    current_time = 0.0
    for msg in midi_file:
        current_time += msg.time
        if not msg.is_meta:
            midi_messages.append((msg.bytes(), current_time))
        if current_time > MIDI_DURATION_THRESHOLD:
            break
    
    return midi_messages



def render_audio(serum, midi_messages, audio_output_file_path):
    duration = midi_messages[-1][1] + 3.0
    audio = serum.process(
        midi_messages=midi_messages,
        duration=duration,
        sample_rate=SAMPLE_RATE,
        num_channels=2,
        buffer_size=512,
        reset=True
    )
    
    # 5. Save Output Audio
    with AudioFile(audio_output_file_path, "w", SAMPLE_RATE, 2) as f:
        f.write(audio)
        print("-> Rendered audio saved to: ", audio_output_file_path)
    

if __name__ == "__main__":
    # Get VST Preset
    vstpreset_paths = get_all_vstpresets(VSTPRESET_DIR, stem=STEM)
    
    # Pre-load all VST Presets
    serum_dict = get_all_serum_states(vstpreset_paths, length=len(vstpreset_paths))
    
    
    # Get MIDI Messages
    midi_file_paths = get_all_midi_files(MIDI_DIR, year=YEAR)
    print(f"Having {len(vstpreset_paths)} vstpresets in stem {STEM}, with {len(midi_file_paths)} midi files in year {YEAR}")
    
    
    
    # Render Audio
    for midi_file_path in tqdm(midi_file_paths, desc="Rendering Audio from Midi Files"):
        
        start_time = time.time()
        midi_messages = get_midi_messages(midi_file_path)
        parsed_midi_file_name = midi_file_path.split('MIDI-Unprocessed_')[-1].split('.midi')[0]
        
        # Load from presets
        for preset_name, serum in serum_dict.items():
            parsed_preset_name = preset_name.split('/')[-1].split('.vstpreset')[0].split('- ')[-1]
            audio_output_file_path = f"{stem_mapping[STEM]}.{parsed_preset_name}.{parsed_midi_file_name}.wav"

            render_audio(
                serum, 
                midi_messages, 
                os.path.join(RENDERED_AUDIO_DIR, audio_output_file_path)
            )

    