import dawdreamer as daw
import numpy as np
import soundfile as sf
import os
from scipy.io import wavfile
import pretty_midi

SAMPLE_RATE = 44100
BLOCK_SIZE  = 512
BPM = 120
RENDER_DURATION = 10
EQ_PLUGIN_PATH = "/Library/Audio/Plug-Ins/VST3/FabFilter Pro-Q 3.vst3"
INPUT_WAV   = "/Users/buffettliu/Desktop/Music_AI/Codes/render_dataset/src/LD - Dat Bit_145008.wav"
OUTPUT_WAV  = "output_with_eq.wav"
EQ_PRESET_PATH = "/Users/buffettliu/Library/Audio/Presets/FabFilter/FabFilter Pro-Q 3/Phone.ffp"
SERUM_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"
SERUM_PRESET_PATH = "../fxp_preset/lead/LD - Dat Bit.fxp"
MIDI_PATH = "../midi/midi_files/train/midi/77964.mid"


if __name__ == "__main__":
    # 1) Engine
    engine = daw.RenderEngine(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    engine.set_bpm(BPM)

    # 2) Load serum plugin
    synth = engine.make_plugin_processor("synth", SERUM_PATH)
    synth.load_preset(SERUM_PRESET_PATH)
    engine.load_graph([
        (synth, [])
    ])
    
    
    # 3) Load MIDI notes
    midi_data = pretty_midi.PrettyMIDI(MIDI_PATH)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            duration = note.end - note.start
            pitch = note.pitch
            velocity = note.velocity if note.velocity > 80 else 80
            synth.add_midi_note(pitch, velocity, start_time, duration)

    engine.render(RENDER_DURATION)
    audio = engine.get_audio()
    wavfile.write('demo.wav', SAMPLE_RATE, audio.transpose())
    synth.clear_midi()
    
    
    # ------------------------------
    # 1) Engine
    engine = daw.RenderEngine(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    engine.set_bpm(BPM)


    # 2) Load synth
    synth = engine.make_plugin_processor("synth", SERUM_PATH)
    synth.load_preset(SERUM_PRESET_PATH)
    
    # 3) Load effects
    eq = engine.make_plugin_processor("fabfilter_eq", EQ_PLUGIN_PATH)
    eq.load_preset(EQ_PRESET_PATH)
    
    # 4) Load graph
    engine.load_graph([
        (synth, []),
        (eq, [synth.get_name()])
    ])
    
    engine.render(RENDER_DURATION)
    audio = engine.get_audio()
    wavfile.write('demo_eq.wav', SAMPLE_RATE, audio.transpose())
    synth.clear_midi()