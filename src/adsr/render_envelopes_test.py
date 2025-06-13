import numpy as np, soundfile as sf, json, random, uuid
from pathlib import Path
from pedalboard import Pedalboard, Gain
from tqdm import trange
from util import ADSR_DICT, SAMPLE_RATE, ms_to_samples, samples_to_ms
import matplotlib.pyplot as plt


BASE_WAV = "../../rendered_one_shot/T20_C4.wav"
stem = "pluck"

OUT_DIR = Path("adsr_pedalboard")
OUT_DIR.mkdir(exist_ok=True)



def adsr_env(total_len, a, d, s_level, r):
    """return total_len (samples) ADSR envelope"""
    def lin(start, end, n):
        return np.linspace(start, end, n, endpoint=False)

    env = np.concatenate([
        lin(0, 1, a),
        lin(1, s_level, d),
        np.full(total_len - (a + d + r), s_level),
        lin(s_level, 0, r)
    ])
    return env


def plot_adsr(env, A, D, H, S, R, signal, original_audio, save_path=None):
    """Plot ADSR envelope, processed waveform, and original audio"""
    fig, (ax3, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1])
    
    # Calculate total length in ms for consistent x-axis
    total_length_ms = samples_to_ms(signal.shape[1])
    
    # Plot ADSR envelope
    time_ms = np.linspace(0, total_length_ms, len(env))
    ax1.plot(time_ms, env)
    
    # Add vertical lines for section boundaries
    ax1.axvline(x=A, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=A+D, color='g', linestyle='--', alpha=0.5)
    ax1.axvline(x=A+D+H, color='b', linestyle='--', alpha=0.5)
    
    # Add labels
    ax1.text(A/2, 0.1, f'Attack\n{A}ms', ha='center')
    ax1.text(A+D/2, 0.1, f'Decay\n{D}ms', ha='center')
    ax1.text(A+D+H/2, 0.1, f'Hold\n{H}ms', ha='center')
    ax1.text(A+D+H+R/2, 0.1, f'Release\n{R}ms', ha='center')
    
    ax1.set_title(f'ADSR Envelope (Sustain Level: {S:.2f})')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_length_ms)
    
    # Plot processed waveform
    time_ms_wave = np.linspace(0, total_length_ms, signal.shape[1])
    ax2.plot(time_ms_wave, signal[0], label='Left Channel', alpha=0.7)
    ax2.plot(time_ms_wave, signal[1], label='Right Channel', alpha=0.7)
    ax2.axvline(x=A, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=A+D, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=A+D+H, color='b', linestyle='--', alpha=0.5)
    ax2.set_title('Processed Waveform')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, total_length_ms)
    
    # Plot original audio
    time_ms_orig = np.linspace(0, total_length_ms, original_audio.shape[1])
    ax3.plot(time_ms_orig, original_audio[0], label='Left Channel', alpha=0.7)
    ax3.plot(time_ms_orig, original_audio[1], label='Right Channel', alpha=0.7)
    ax3.axvline(x=A, color='r', linestyle='--', alpha=0.5)
    ax3.axvline(x=A+D, color='g', linestyle='--', alpha=0.5)
    ax3.axvline(x=A+D+H, color='b', linestyle='--', alpha=0.5)
    ax3.set_title('Original Audio')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, total_length_ms)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

meta = []
audio, sr = sf.read(BASE_WAV)
audio = audio.T
audio = audio[:, 44100:] # 3 secs
assert sr == SAMPLE_RATE
note = np.tile(audio, 1)



for i in trange(10):
    # Random parameters in milliseconds
    A = random.randint(ADSR_DICT[stem]["a"][0], ADSR_DICT[stem]["a"][1])              # Attack: 0-250 ms
    D = random.randint(ADSR_DICT[stem]["d"][0], ADSR_DICT[stem]["d"][1])             # Decay: 0-1000 ms
    H = random.randint(ADSR_DICT[stem]["h"][0], ADSR_DICT[stem]["h"][1])              # Hold: 0-500 ms
    S = round(random.uniform(ADSR_DICT[stem]["s"][0], ADSR_DICT[stem]["s"][1]), 2)    # Sustain: 0.5-1.0
    R = random.randint(ADSR_DICT[stem]["r"][0], ADSR_DICT[stem]["r"][1])            # Release: 300-800 ms
    length = A + D + H + R
    
    # Convert ms to samples
    A_samples = ms_to_samples(A)
    D_samples = ms_to_samples(D)
    H_samples = ms_to_samples(H)
    R_samples = ms_to_samples(R)
    length = A_samples + D_samples + H_samples + R_samples

    env = adsr_env(length, A_samples, D_samples, S, R_samples)
    env = env.reshape(1, -1)  # Reshape to (1, length) for broadcasting

    signal = note[:, :length] * env   # Now shapes will be (2, length) * (1, length)
    
    # Plot ADSR envelope and waveforms
    plot_adsr(env[0], A, D, H, S, R, signal, note[:, :length], save_path=OUT_DIR / f"{i}_adsr.png")
    
    # Pedalboard 例：加點輸出增益
    board = Pedalboard([Gain(gain_db=0.0)])
    processed = board(signal, SAMPLE_RATE)

    fname = f"{i}.wav"
    sf.write(OUT_DIR / fname, processed.T, SAMPLE_RATE)  # Transpose back to (samples, channels)

    meta.append({
        "file": fname,
        "stem": stem,
        "attack": A,   # Already in ms
        "decay": D,    # Already in ms
        "hold": H,     # Already in ms
        "sustain": S,
        "release": R   # Already in ms
    })


with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=4, ensure_ascii=False)