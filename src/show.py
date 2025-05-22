import soundfile as sf

from pprint import pprint
from pedalboard import load_plugin 
from mido import MidiFile, Message
from pedalboard.io import AudioFile

SERUM_PLUGIN_PATH = "/Library/Audio/Plug-Ins/Components/Serum.component" # "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
PLUGIN_NAME = "Serum" #"Serum 2" # "Serum 2 FX"
# FXP_PRESET_PATH = "serum preset/lead/LD - Starboy Era.fxp" 
MIDI_INPUT_FILE_PATH = "../midi/piano.mid"  # <-- CHANGE THIS
AUDIO_OUTPUT_FILE_PATH = "output_serum_audio.wav" # <-- CHANGE THIS
PRESET_DATA_FILE_PATH = "jsons/serum_parameters2.json"
SAMPLE_RATE = 44100.0  # Hz
BUFFER_SIZE = 512      # Samples per processing block (power of 2 often good)
OUTPUT_CHANNELS = 2    # 1 for mono, 2 for stereo (Serum often outputs stereo)
TAIL_DURATION_SECONDS = 3.0 # Extra time to capture release tails after MIDI ends



if __name__ == "__main__":
    # serum = load_plugin("/Library/Audio/Plug-Ins/VST3/Serum2.vst3", plugin_name="Serum 2")
    serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)
        
    with open('../vstpreset/lead/LD - Arp Attack.vstpreset', 'rb') as f:
        serum.raw_state = f.read()

    # Load the MIDI file
    midi_file = MidiFile(MIDI_INPUT_FILE_PATH)
    midi_messages = []
    current_time = 0.0
    for msg in midi_file:
        current_time += msg.time
        if not msg.is_meta:
            midi_messages.append((msg.bytes(), current_time))

    
    # 4. Render Audio
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
    with AudioFile("output.wav", "w", SAMPLE_RATE, 2) as f:
        f.write(audio)
        print("Audio saved to output.wav")