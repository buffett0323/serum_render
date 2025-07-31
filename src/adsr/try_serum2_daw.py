import logging
import os
import json
import random
import dawdreamer as daw

from pathlib import Path
from glob import glob
from scipy.io import wavfile
from tqdm import tqdm


bpm = 120
sample_rate = 44100
plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
preset_path = "../../fxp_preset/serum1/Albert_Serum1/KEY_01_4am.fxp" #"../../fxp_preset/serum2/Lead/LD - Analog Glow.SerumPreset"


note_number = 60
velocity = 100
time_seconds = 0
note_duration = 4
render_duration = note_duration + 2

print(os.path.exists(preset_path))

engine = daw.RenderEngine(sample_rate, block_size=512)
engine.set_bpm(bpm)

# Create synth processor
synth = engine.make_plugin_processor("synth", plugin_path)
graph = [(synth, [])]
engine.load_graph(graph)

synth.load_preset(preset_path)

synth.add_midi_note(note_number, velocity, time_seconds, note_duration)

engine.render(render_duration)
audio = engine.get_audio().mean(axis=0)

output_filename = "Serum2_test.wav"
wavfile.write(str(output_filename), sample_rate, audio)