from pedalboard import load_plugin
from pedalboard.io import AudioFile
from mido import MidiFile
import json
from pprint import pprint

# 1. 配置路径
VST3_PATH    = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
PRESET_PATH  = "serum preset/bass/BA - Arp It Up.fxp"
MIDI_PATH    = "piano.mid"
OUTPUT_WAV   = "output_serum_audio.wav"
PLUGIN_NAME  = "Serum 2"
SAMPLE_RATE  = 44100


# 3. 加载预设（仅支持 .vstpreset 格式）
def load_fxp_preset(serum_instance, fxp_path):
    with open(fxp_path, "rb") as f:
        serum_instance.state = f.read()
        

if __name__ == "__main__":
    # Load Serum plugin
    serum = load_plugin(VST3_PATH, plugin_name=PLUGIN_NAME)
    print(f"Successfully loaded Serum: {serum.name}")
    
    
    # Load preset
    # load_fxp_preset(serum, PRESET_PATH)


    
    mid = MidiFile(MIDI_PATH)
    midi_messages = []
    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta:
                midi_messages.append(msg)


    duration = mid.length
    audio = serum.process(
        midi_messages,
        duration=duration,
        sample_rate=SAMPLE_RATE
    )                                                                #  [oai_citation:2‡GitHub](https://github.com/spotify/pedalboard)

    # Store
    with AudioFile(OUTPUT_WAV, 'w', SAMPLE_RATE, audio.shape[0]) as f:
        f.write(audio)                                               #  [oai_citation:3‡GitHub](https://github.com/spotify/pedalboard)
        
        
