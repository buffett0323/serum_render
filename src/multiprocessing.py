import os
import time
import soundfile as sf

from tqdm import tqdm
from pprint import pprint
from pedalboard import load_plugin 
from mido import MidiFile, Message
from pedalboard.io import AudioFile
from multiprocessing import Pool, cpu_count

from process import get_midi_messages, get_all_vstpresets, get_all_midi_files, render_audio, stem_mapping


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


def render_task(args):
    vstpreset_path, midi_file_path = args
    try:
        # Load Serum
        serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)

        with open(vstpreset_path, 'rb') as f:
            serum.raw_state = f.read()

        # Get MIDI
        midi_messages = get_midi_messages(midi_file_path)

        parsed_preset_name = os.path.basename(vstpreset_path).split('.vstpreset')[0].split('- ')[-1]
        parsed_midi_file_name = midi_file_path.split('MIDI-Unprocessed_')[-1].split('.midi')[0]
        audio_output_file_path = f"{stem_mapping[STEM]}.{parsed_preset_name}.{parsed_midi_file_name}.wav"
        full_audio_output_path = os.path.join(RENDERED_AUDIO_DIR, audio_output_file_path)

        render_audio(serum, midi_messages, full_audio_output_path)
        return f"✅ Done: {audio_output_file_path}"
    
    except Exception as e:
        return f"❌ Error with {vstpreset_path} & {midi_file_path}: {e}"


if __name__ == "__main__":
    vstpreset_paths = get_all_vstpresets(VSTPRESET_DIR, stem=STEM)
    midi_file_paths = get_all_midi_files(MIDI_DIR, year=YEAR)

    print(f"Having {len(vstpreset_paths)} vstpresets in stem {STEM}, with {len(midi_file_paths)} midi files in year {YEAR}")

    # Create task list: all combinations
    task_list = []
    for vst_path in vstpreset_paths:
        for midi_path in midi_file_paths:
            task_list.append((vst_path, midi_path))

    print(f"Total render tasks: {len(task_list)}")

    with Pool(processes=min(cpu_count(), 8)) as pool:
        for result in tqdm(pool.imap_unordered(render_task, task_list), total=len(task_list), desc="Rendering"):
            print(result)