import os
import time
import mido
import soundfile as sf

from tqdm import tqdm
from pprint import pprint
from pedalboard import load_plugin 
from pedalboard.io import AudioFile
from mido import MidiFile, Message, merge_tracks, tick2second

from visualize_midi import plot_midi_piano_roll

SERUM_PLUGIN_PATH = os.path.expanduser("/Users/bliu/Library/Audio/Plug-Ins/Components/Serum.component") #"/Library/Audio/Plug-Ins/Components/Serum.component" #os.path.expanduser("/Users/bliu/Library/Audio/Plug-Ins/Components/Serum.component")
PLUGIN_NAME = "Serum" #"Serum 2" # "Serum 2 FX"
VSTPRESET_DIR = "../vstpreset"
SPLIT = "train" #"evaluation"
MIDI_DIR = f"../midi/midi_files/{SPLIT}/midi/"
RENDERED_AUDIO_DIR = f"../rendered_audio/{SPLIT}"
SAMPLE_RATE = 44100.0  # Hz
BUFFER_SIZE = 512      # Samples per processing block (power of 2 often good)
OUTPUT_CHANNELS = 2
TAIL_DURATION_SECONDS = 2.0  # Extra time to capture release tails after MIDI ends
MIDI_DURATION_THRESHOLD = 10.0 # 3 minutes
STEM = "keys"

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


def get_all_midi_files(midi_dir):
    return [
        os.path.join(midi_dir, f)
        for f in os.listdir(midi_dir)
        if f.endswith(".mid")
    ]


def get_all_serum_states(vstpreset_paths):
    serum_dict = {}
    for preset_path in tqdm(vstpreset_paths, desc="Loading VST Presets"):
        serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)
        with open(preset_path, 'rb') as f:
            serum.raw_state = f.read()
        serum_dict[preset_path] = serum
    return serum_dict


def get_midi_messages(midi_file_path):
    midi_file = MidiFile(midi_file_path)
    ticks_per_beat = midi_file.ticks_per_beat

    tempo = 500000  # Default 120 BPM
    absolute_time = 0  # in ticks
    time_in_seconds = 0.0
    messages = []
    first_note_time = None

    merged = merge_tracks(midi_file.tracks)

    for msg in merged:
        absolute_time += msg.time
        delta_seconds = tick2second(msg.time, ticks_per_beat, tempo)
        time_in_seconds += delta_seconds

        if msg.type == 'set_tempo':
            tempo = msg.tempo
            continue

        if msg.type == 'note_on' and msg.velocity == 0:
            msg = Message('note_off', note=msg.note, velocity=0, channel=msg.channel)

        if msg.type in ['note_on', 'note_off']:
            if msg.type == 'note_on' and first_note_time is None:
                first_note_time = time_in_seconds
            msg.time = time_in_seconds
            messages.append(msg)

    if first_note_time is None:
        first_note_time = -1

    return messages, first_note_time


def render_audio(serum, midi_messages, first_note_time, audio_output_file_path):
    duration = MIDI_DURATION_THRESHOLD # midi_messages[-1].time + TAIL_DURATION_SECONDS
    audio = serum.process(
        midi_messages=midi_messages,
        duration=duration,
        sample_rate=SAMPLE_RATE,
        num_channels=2,
        buffer_size=BUFFER_SIZE,
        reset=True
    )

    start_sample = int(first_note_time * SAMPLE_RATE)
    if first_note_time >= 0:
        audio = audio[:, start_sample:]

    with AudioFile(audio_output_file_path, "w", SAMPLE_RATE, OUTPUT_CHANNELS) as f:
        f.write(audio)
        print("-> Rendered audio saved to:", audio_output_file_path)


if __name__ == "__main__":
    # Get VST Presets
    vstpreset_paths = get_all_vstpresets(VSTPRESET_DIR, stem=STEM) #[-2:]
    serum_dict = get_all_serum_states(vstpreset_paths)

    # Get Midi Files
    # midi_file_paths = get_all_midi_files(MIDI_DIR)[:100]
    # print(f"Having {len(vstpreset_paths)} vstpresets in stem {STEM}, with {len(midi_file_paths)} midi files")

    # with open(f"../info/{SPLIT}_midi_file_paths.txt", "w") as f:
    #     for midi_file_path in midi_file_paths:
    #         f.write(midi_file_path + "\n")
    with open(f"../info/{SPLIT}_midi_file_paths.txt", "r") as f:
        midi_file_paths = [line.strip() for line in f.readlines()]

    # Render Audio from Midi Files
    for midi_file_path in tqdm(midi_file_paths, desc="Rendering Audio from Midi Files"):
        midi_messages, first_note_time = get_midi_messages(midi_file_path)
        parsed_midi_file_name = os.path.basename(midi_file_path).replace(".mid", "")

        for preset_path, serum in serum_dict.items():
            preset_name = os.path.basename(preset_path).split("-")[-1].replace(".vstpreset", "")
            output_filename = f"{stem_mapping[STEM]}.{preset_name}.{parsed_midi_file_name}.wav"
            output_path = os.path.join(RENDERED_AUDIO_DIR, output_filename)

            render_audio(serum, midi_messages, first_note_time, output_path)