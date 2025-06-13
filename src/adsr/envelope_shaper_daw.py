import dawdreamer as daw
import numpy as np, soundfile as sf, json, random, uuid
from pathlib import Path
from tqdm import trange


out_dir = Path("dd_adsr_dataset")
out_dir.mkdir(exist_ok=True)


SR, BLOCK = 44100, 512
N_DATA = 200
engine = daw.RenderEngine(SR, BLOCK)

# ---------- sample ---------- #
sample, sr = sf.read("../rendered_one_shot/T20_C4.wav")
sample = sample.T  # Convert to [channels, samples]
sample = sample[:, 44100:88200]
print("Sample shape:", sample.shape)
sf.write(out_dir / "sample.wav", sample.T, SR)


sampler = engine.make_sampler_processor("sampler", sample)
sampler.record = True

graph = [(sampler, [])]
engine.load_graph(graph)

# ---------- batch render ---------- #
meta = []

# First, let's see all available parameters and their current values
print("\nBefore setting parameters:")
param_names_to_idx = {}
for i in range(sampler.get_parameter_size()):
    name = sampler.get_parameter_name(i)
    value = sampler.get_parameter(i)
    text = sampler.get_parameter_text(i)
    param_names_to_idx[name] = i
    print(f"{name}: value={value}, text={text}")

# Set ADSR parameters
A = 2000  # Attack in ms (2 seconds)
D = 2000  # Decay in ms (2 seconds)
S = 1.0   # Sustain (100%)
R = 2000  # Release in ms (2 seconds)

print("\nSetting parameters to:")
print(f"Attack: {A}ms")
print(f"Decay: {D}ms")
print(f"Sustain: {S}")
print(f"Release: {R}ms")

# Set the parameters
sampler.set_parameter(param_names_to_idx["Amp Env Attack"], A)
sampler.set_parameter(param_names_to_idx["Amp Env Decay"], D)
sampler.set_parameter(param_names_to_idx["Amp Env Sustain"], S)
sampler.set_parameter(param_names_to_idx["Amp Env Release"], R)

# Verify the changes
print("\nAfter setting parameters:")
for i in range(sampler.get_parameter_size()):
    name = sampler.get_parameter_name(i)
    value = sampler.get_parameter(i)
    text = sampler.get_parameter_text(i)
    print(f"{name}: value={value}, text={text}")

# Test with a note
sampler.clear_midi()
sampler.add_midi_note(note=60, velocity=100, start_time=0.0, duration=1.0)

# Make sure we're recording
sampler.record = True
engine.render(2.0)

# Get audio from the sampler, not the engine
audio = sampler.get_audio()      # [ch, samples]
print("\nAudio shape:", audio.shape)

# Save both the sampler output and engine output for comparison
fname = "test_adsr.wav"
sf.write(out_dir / fname, audio.T, SR)
print(f"\nSaved sampler output to: {fname}")

# Also save engine output for comparison
engine_audio = engine.get_audio()
sf.write(out_dir / "test_engine.wav", engine_audio.T, SR)
print(f"Saved engine output to: test_engine.wav")

# meta.append(dict(
#     file=fname,
#     attack_ms=A, 
#     decay_ms=D,
#     sustain=S, 
#     release_ms=R
# ))

# with open(out_dir / "metadata.json", "w") as f:
#     json.dump(meta, f, indent=4, ensure_ascii=False)