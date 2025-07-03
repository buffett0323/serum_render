import json
import numpy as np
from librosa.effects import pitch_shift
import pretty_midi
import soundfile as sf
import matplotlib.pyplot as plt

one_shot_path = "/mnt/gestalt/home/buffett/adsr/rendered_one_shot/T354_C4.wav"
midi_path = "/mnt/gestalt/home/buffett/EDM_FAC_DATA/single_note_midi/evaluation/midi/1815011.mid"
adsr_path = "stats/envelopes_train_new.json"

SAMPLE_RATE = 44100
TOTAL_DURATION = 3
REFERENCE_MIDI_NOTE = 60  # C4

# Load the timbre audio
timbre, sr = sf.read(one_shot_path)
if timbre.ndim > 1:
    timbre = timbre[:, 0]  # Convert to mono if stereo

# Resample if necessary
if sr != SAMPLE_RATE:
    from librosa.core import resample
    timbre = resample(timbre, orig_sr=sr, target_sr=SAMPLE_RATE)

with open(adsr_path, "r") as f:
    adsr_bank = json.load(f)

print(adsr_bank[0])

def pitch_shift_audio(one_shot: np.ndarray, n_steps: float) -> np.ndarray:
    # C4 = MIDI note 60, so if note.pitch is 62 (D4), pitch_shift = 2 semitones up
    # If note.pitch is 58 (Bb3), pitch_shift = -2 semitones down
    return pitch_shift(one_shot, sr=SAMPLE_RATE, n_steps=n_steps)

def process_adsr(adsr):
    A = adsr["attack"]
    D = adsr["decay"]
    H = adsr["hold"]
    S = adsr["sustain"]
    R = adsr["release"]

    def ms_to_samples(ms):
        return int(ms * SAMPLE_RATE / 1000)

    A_samples = ms_to_samples(A)
    D_samples = ms_to_samples(D)
    H_samples = ms_to_samples(H)
    R_samples = ms_to_samples(R)
    length = A_samples + D_samples + H_samples + R_samples
    return length, A_samples, D_samples, S, R_samples

midi = pretty_midi.PrettyMIDI(midi_path)
total_samples = int(np.ceil(TOTAL_DURATION * SAMPLE_RATE))

# Use first instrument that has notes; otherwise skip
instruments = [inst for inst in midi.instruments if inst.notes]
if not instruments:
    # Return empty audio for all ADSR envelopes
    print("No instruments with notes found")
    exit()

inst = instruments[0]

# Preload all pitch-shifted timbres for this MIDI file
# print(f"Preloading pitch-shifted timbres for {midi_path}...")
pitch_shifted_timbres = {}

# Get unique pitches in this MIDI file
unique_pitches = set()
for note in inst.notes:
    if note.start < TOTAL_DURATION:
        unique_pitches.add(note.pitch)

# Preload pitch-shifted timbres for all unique pitches
for pitch in unique_pitches:
    pitch_shift_steps = pitch - REFERENCE_MIDI_NOTE
    pitch_shifted_timbres[pitch] = pitch_shift_audio(timbre, pitch_shift_steps)

# print(f"Preloaded {len(pitch_shifted_timbres)} pitch-shifted timbres")

# Render for each ADSR envelope
results = []
adsr = adsr_bank[0]
mix_with_env = np.zeros(total_samples, dtype=np.float32)
mix_without_env = np.zeros(total_samples, dtype=np.float32)

# Process each note
for note in inst.notes:
    if note.start >= TOTAL_DURATION:
        break
    
    start_samp = int(note.start * SAMPLE_RATE)
    note_dur_samp = int((note.end - note.start) * SAMPLE_RATE)
    note_dur_samp = max(note_dur_samp, 1)  # safety

    # Get preloaded pitch-shifted timbre
    seg = pitch_shifted_timbres[note.pitch]
    
    # Process ADSR data
    length, A_samples, D_samples, S, R_samples = process_adsr(adsr)
    
    # Create envelope for the full note duration (including release)
    total_env_samples = note_dur_samp + R_samples
    env = np.zeros(total_env_samples, dtype=np.float32)
    
    # Attack phase (0 to 1)
    if A_samples > 0:
        attack_end = min(A_samples, total_env_samples)
        env[:attack_end] = np.linspace(0, 1, attack_end, endpoint=False)
    
    # Decay phase (1 to sustain level)
    if D_samples > 0 and A_samples < total_env_samples:
        decay_start = A_samples
        decay_end = min(decay_start + D_samples, total_env_samples)
        if decay_end > decay_start:
            env[decay_start:decay_end] = np.linspace(1, S, decay_end - decay_start, endpoint=False)
    
    # Sustain phase (sustain level)
    sustain_start = min(A_samples + D_samples, total_env_samples)
    sustain_end = min(sustain_start + (total_env_samples - sustain_start - R_samples), total_env_samples)
    if sustain_end > sustain_start:
        env[sustain_start:sustain_end] = S
    
    # Release phase (sustain to 0) - starts when note ends
    if R_samples > 0 and note_dur_samp < total_env_samples:
        release_start = note_dur_samp
        release_end = min(release_start + R_samples, total_env_samples)
        if release_end > release_start:
            env[release_start:release_end] = np.linspace(S, 0, release_end - release_start, endpoint=False)

    # Apply envelope to timbre (only for the note duration)
    actual_note_dur = min(note_dur_samp, len(seg))
    note_audio_with_env = seg[:actual_note_dur] * env[:actual_note_dur]
    note_audio_without_env = seg[:actual_note_dur]  # No envelope applied

    # Mix into buffer (additive) - only the note part
    end_samp = min(start_samp + actual_note_dur, total_samples)
    if end_samp > start_samp:
        mix_with_env[start_samp:end_samp] += note_audio_with_env[:end_samp - start_samp]
        mix_without_env[start_samp:end_samp] += note_audio_without_env[:end_samp - start_samp]
    
    # Handle release phase separately - it continues after note ends
    if R_samples > 0 and actual_note_dur < total_env_samples:
        release_start = actual_note_dur
        release_end = min(release_start + R_samples, total_env_samples)
        if release_end > release_start:
            # For release, we need to continue the timbre (could be silence or last sample)
            release_timbre = np.full(release_end - release_start, seg[-1] if len(seg) > 0 else 0)
            release_env = env[release_start:release_end]
            release_audio = release_timbre * release_env
            
            # Add release to the mix buffer
            release_start_samp = start_samp + actual_note_dur
            release_end_samp = min(release_start_samp + len(release_audio), total_samples)
            if release_end_samp > release_start_samp:
                mix_with_env[release_start_samp:release_end_samp] += release_audio[:release_end_samp - release_start_samp]

# Simple peak-normalise to -1..1 (prevents clipping)
max_amp_with_env = np.abs(mix_with_env).max() + 1e-9
max_amp_without_env = np.abs(mix_without_env).max() + 1e-9
mix_with_env /= max_amp_with_env
mix_without_env /= max_amp_without_env

results.append(mix_with_env)

# Create time axis for plotting
time_axis = np.linspace(0, TOTAL_DURATION, total_samples)

# Create the plot
plt.figure(figsize=(15, 12))

# Plot 1: Original track (without envelope)
plt.subplot(4, 1, 1)
plt.plot(time_axis, mix_without_env, 'b-', linewidth=0.5, alpha=0.8)
plt.title('Original Track (Without ADSR Envelope)', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, TOTAL_DURATION)

# Plot 2: Track with envelope
plt.subplot(4, 1, 2)
plt.plot(time_axis, mix_with_env, 'r-', linewidth=0.5, alpha=0.8)
plt.title('Track with ADSR Envelope Applied', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, TOTAL_DURATION)

# Plot 3: ADSR Envelope Curve
plt.subplot(4, 1, 3)
# Create a time axis for the envelope (longer to show full release)
max_env_time = TOTAL_DURATION + 2.0  # Add extra time for release
env_time_axis = np.linspace(0, max_env_time, int(max_env_time * SAMPLE_RATE))

# Create the ADSR envelope for visualization
env_vis = np.zeros(len(env_time_axis), dtype=np.float32)
length, A_samples, D_samples, S, R_samples = process_adsr(adsr)

# Attack phase (0 to 1)
if A_samples > 0:
    attack_end = min(A_samples, len(env_vis))
    env_vis[:attack_end] = np.linspace(0, 1, attack_end, endpoint=False)

# Decay phase (1 to sustain level)
if D_samples > 0 and A_samples < len(env_vis):
    decay_start = A_samples
    decay_end = min(decay_start + D_samples, len(env_vis))
    if decay_end > decay_start:
        env_vis[decay_start:decay_end] = np.linspace(1, S, decay_end - decay_start, endpoint=False)

# Sustain phase (sustain level) - extend for visualization
sustain_start = min(A_samples + D_samples, len(env_vis))
# For visualization, let's have sustain last for a reasonable time before release
sustain_duration = min(len(env_vis) - sustain_start - R_samples, int(1.0 * SAMPLE_RATE))  # 1 second of sustain
sustain_end = sustain_start + sustain_duration
if sustain_duration > 0:
    env_vis[sustain_start:sustain_end] = S

# Release phase (sustain to 0) - starts from sustain level
if R_samples > 0:
    release_start = sustain_end
    release_end = min(release_start + R_samples, len(env_vis))
    if release_end > release_start:
        env_vis[release_start:release_end] = np.linspace(S, 0, release_end - release_start, endpoint=False)

plt.plot(env_time_axis, env_vis, 'g-', linewidth=2, alpha=0.8, label='ADSR Envelope')

# Add phase annotations
if A_samples > 0:
    attack_mid = A_samples / 2 / SAMPLE_RATE
    plt.annotate('Attack', xy=(attack_mid, 0.5), xytext=(attack_mid, 0.7), 
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center')

if D_samples > 0:
    decay_mid = (A_samples + D_samples/2) / SAMPLE_RATE
    plt.annotate('Decay', xy=(decay_mid, (1+S)/2), xytext=(decay_mid, 0.8), 
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center')

sustain_mid = (sustain_start + sustain_duration/2) / SAMPLE_RATE
plt.annotate('Sustain', xy=(sustain_mid, S), xytext=(sustain_mid, S+0.1), 
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center')

if R_samples > 0:
    release_mid = (release_start + R_samples/2) / SAMPLE_RATE
    plt.annotate('Release', xy=(release_mid, S/2), xytext=(release_mid, S+0.2), 
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center')

plt.title('ADSR Envelope Curve', fontsize=14, fontweight='bold')
plt.ylabel('Envelope Value', fontsize=12)
plt.xlabel('Time (seconds)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, max_env_time)
plt.ylim(0, 1.1)
plt.legend()

# Add text box with ADSR parameters
adsr_text = f'Attack: {adsr["attack"]:.1f}ms\nDecay: {adsr["decay"]:.1f}ms\nSustain: {adsr["sustain"]:.3f}\nRelease: {adsr["release"]:.1f}ms'
plt.text(0.02, 0.98, adsr_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Both tracks overlaid for comparison with ADSR envelope
plt.subplot(4, 1, 4)

# Create a composite envelope that shows the ADSR curve for each note
composite_env = np.zeros(total_samples, dtype=np.float32)

# Process each note to create the composite envelope
for note in inst.notes:
    if note.start >= TOTAL_DURATION:
        break
    
    start_samp = int(note.start * SAMPLE_RATE)
    note_dur_samp = int((note.end - note.start) * SAMPLE_RATE)
    note_dur_samp = max(note_dur_samp, 1)  # safety
    
    # Create envelope for this note
    total_env_samples = note_dur_samp + R_samples
    env = np.zeros(total_env_samples, dtype=np.float32)
    
    # Attack phase (0 to 1)
    if A_samples > 0:
        attack_end = min(A_samples, total_env_samples)
        env[:attack_end] = np.linspace(0, 1, attack_end, endpoint=False)
    
    # Decay phase (1 to sustain level)
    if D_samples > 0 and A_samples < total_env_samples:
        decay_start = A_samples
        decay_end = min(decay_start + D_samples, total_env_samples)
        if decay_end > decay_start:
            env[decay_start:decay_end] = np.linspace(1, S, decay_end - decay_start, endpoint=False)
    
    # Sustain phase (sustain level)
    sustain_start = min(A_samples + D_samples, total_env_samples)
    sustain_end = min(sustain_start + (total_env_samples - sustain_start - R_samples), total_env_samples)
    if sustain_end > sustain_start:
        env[sustain_start:sustain_end] = S
    
    # Release phase (sustain to 0) - starts when note ends
    if R_samples > 0 and note_dur_samp < total_env_samples:
        release_start = note_dur_samp
        release_end = min(release_start + R_samples, total_env_samples)
        if release_end > release_start:
            env[release_start:release_end] = np.linspace(S, 0, release_end - release_start, endpoint=False)

    # Add this note's envelope to the composite
    end_samp = min(start_samp + total_env_samples, total_samples)
    if end_samp > start_samp:
        composite_env[start_samp:end_samp] = np.maximum(composite_env[start_samp:end_samp], 
                                                       env[:end_samp - start_samp])

# Plot the audio tracks
plt.plot(time_axis, mix_without_env, 'b-', linewidth=0.5, alpha=0.7, label='Without Envelope')
plt.plot(time_axis, mix_with_env, 'r-', linewidth=0.5, alpha=0.7, label='With Envelope')

# Plot the ADSR envelope overlaid (scaled to fit in the plot)
env_scale = 0.8  # Scale factor to make envelope visible
plt.plot(time_axis, composite_env * env_scale, 'g-', linewidth=2, alpha=0.9, label='ADSR Envelope (scaled)')

# Add vertical lines to show note boundaries
for note in inst.notes:
    if note.start < TOTAL_DURATION:
        plt.axvline(x=note.start, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=min(note.end, TOTAL_DURATION), color='orange', linestyle='--', alpha=0.5, linewidth=1)

plt.title('Comparison: Original vs Envelope Applied with ADSR Curve Overlaid', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, TOTAL_DURATION)

plt.tight_layout()

# Save the plot
output_filename = 'adsr_plot.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_filename}")

# Save the audio files
import soundfile as sf

# Save original audio (without envelope)
original_audio_filename = 'original_audio_without_env.wav'
sf.write(original_audio_filename, mix_without_env, SAMPLE_RATE)
print(f"Original audio (without envelope) saved as: {original_audio_filename}")

# Save audio with envelope applied
envelope_audio_filename = 'audio_with_adsr_env.wav'
sf.write(envelope_audio_filename, mix_with_env, SAMPLE_RATE)
print(f"Audio with ADSR envelope saved as: {envelope_audio_filename}")

# Print some statistics
print(f"\nAudio Statistics:")
print(f"Original track - Max amplitude: {np.abs(mix_without_env).max():.4f}")
print(f"Envelope track - Max amplitude: {np.abs(mix_with_env).max():.4f}")
print(f"RMS difference: {np.sqrt(np.mean((mix_with_env - mix_without_env)**2)):.4f}")

# Print file information
print(f"\nFile Information:")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Duration: {TOTAL_DURATION} seconds")
print(f"Total samples: {total_samples}")
print(f"ADSR parameters used:")
print(f"  Attack: {adsr['attack']:.1f} ms")
print(f"  Decay: {adsr['decay']:.1f} ms")
print(f"  Sustain: {adsr['sustain']:.3f}")
print(f"  Release: {adsr['release']:.1f} ms")