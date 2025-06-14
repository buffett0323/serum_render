import numpy as np, soundfile as sf, json, random, uuid, os
from pathlib import Path
from pedalboard import Pedalboard, Gain
from tqdm import tqdm
from util import SAMPLE_RATE, ms_to_samples, samples_to_ms, SERUM_ADSR
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
START_POINT = 22050
STEMS = ["lead", "keys", "pad", "pluck", "synth", "vox"]
OUT_DIR = Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset")
OUT_DIR.mkdir(exist_ok=True)

def adsr_env(total_len, a, d, s_level, r):
    def lin(start, end, n):
        return np.linspace(start, end, n, endpoint=False)

    env = np.concatenate([
        lin(0, 1, a),
        lin(1, s_level, d),
        np.full(total_len - (a + d + r), s_level),
        lin(s_level, 0, r)
    ])
    return env

def process_timbre_pair(tc_pair, envelopes_metadata, timbre_content_path, out_dir):
    meta_chunk = []
    audio, sr = sf.read(os.path.join(timbre_content_path, tc_pair))
    audio = audio.T
    audio = audio[:, START_POINT:]  # 3.5 secs
    note = np.tile(audio, 1)
    
    for env_data in envelopes_metadata:
        # Random parameters in milliseconds
        ID = env_data["id"]
        STEM = env_data["stem"]
        A = env_data["attack"]
        D = env_data["decay"]
        H = env_data["hold"]
        S = env_data["sustain"]
        R = env_data["release"]
        length = env_data["length"]
        
        # Convert ms to samples
        A_samples = ms_to_samples(A)
        D_samples = ms_to_samples(D)
        H_samples = ms_to_samples(H)
        R_samples = ms_to_samples(R)
        length = A_samples + D_samples + H_samples + R_samples

        env = adsr_env(length, A_samples, D_samples, S, R_samples)
        env = env.reshape(1, -1)  # Reshape to (1, length) for broadcasting

        signal = note[:, :length] * env   # Now shapes will be (2, length) * (1, length)

        # Pedalboard mixing
        board = Pedalboard([Gain(gain_db=0.0)])
        processed = board(signal, SAMPLE_RATE)

        fname = f"ENV_{ID}_{tc_pair.split('.wav')[0]}_{STEM}.wav"
        sf.write(out_dir / fname, processed.T, SAMPLE_RATE)  # Transpose back to (samples, channels)

        meta_chunk.append({
            "file": fname,
            "stem": STEM,
            "attack": A,   # Already in ms
            "decay": D,    # Already in ms
            "hold": H,     # Already in ms
            "sustain": S,
            "release": R   # Already in ms
        })
    
    return meta_chunk


if __name__ == "__main__":
    timbre_content_path = "/mnt/gestalt/home/buffett/rendered_one_shot"
    timbre_content_pairs = []
    with open("stats/chosen_timbre_content_pairs.txt", "r") as f:
        for line in f:
            timbre_content_pairs.append(line.strip())
            
    with open("stats/envelopes_metadata.json", "r") as f:
        envelopes_metadata = json.load(f)
            
    print(f"Timbre-content pairs: {len(timbre_content_pairs)}")
    print(f"Envelopes metadata: {len(envelopes_metadata)}")

    # Create a partial function with the fixed arguments
    process_func = partial(
        process_timbre_pair,
        envelopes_metadata=envelopes_metadata,
        timbre_content_path=timbre_content_path,
        out_dir=OUT_DIR
    )

    # Use multiprocessing to process timbre pairs in parallel
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_func, timbre_content_pairs),
            total=len(timbre_content_pairs),
            desc="Processing timbre pairs"
        ))

    # Flatten the results and save metadata
    meta = [item for sublist in results for item in sublist]
    
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)