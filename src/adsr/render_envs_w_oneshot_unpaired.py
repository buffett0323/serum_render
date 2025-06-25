import numpy as np, soundfile as sf, json, random, os
from pathlib import Path
from pedalboard import Pedalboard, Gain
from tqdm import tqdm
from util import SAMPLE_RATE, ms_to_samples, samples_to_ms, SERUM_ADSR
import multiprocessing as mp
from functools import partial

SAMPLE_RATE = 44100
START_POINT = 22050
AMOUNT = 10
SPLIT = "train" # val # test
STEMS = ["lead", "keys", "pad", "pluck", "synth", "vox"]
BASE_DIR = "/mnt/gestalt/home/buffett" #"/home/buffett/dataset"
OUT_DIR = Path(f"{BASE_DIR}/rendered_adsr_unpaired/{SPLIT}") #"/mnt/gestalt/home/buffett/rendered_adsr_dataset"
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

def process_single_envelope(env_data, tc_pairs, timbre_content_path, out_dir, amount):
    """Process a single envelope with multiple timbre content pairs"""
    meta_chunk = []
    
    # Random parameters in milliseconds
    ID = env_data["id"]
    STEM = env_data["stem"]
    A = env_data["attack"]
    D = env_data["decay"]
    H = env_data["hold"]
    S = env_data["sustain"]
    R = env_data["release"]
    length = env_data["length"]
    
    for amt in range(amount):
        try:
            tc_pair = random.choice(tc_pairs)
            audio, _ = sf.read(os.path.join(timbre_content_path, tc_pair))
            audio = audio.T
            audio = audio[:, START_POINT:]  # 3.5 secs
            note = np.tile(audio, 1)
        
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

            fname = f"ENV_{ID}_{tc_pair.split('.wav')[0]}_{STEM}_{amt}.wav"
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
        except Exception as e:
            print(f"Error processing envelope {ID}, timbre {tc_pair}, amount {amt}: {e}")
            continue
    
    return meta_chunk

def process_timbre_pair_parallel(tc_pairs, envelopes_metadata, timbre_content_path, out_dir, num_processes=None):
    """Process envelopes in parallel using multiprocessing"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} processes")
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_single_envelope,
        tc_pairs=tc_pairs,
        timbre_content_path=timbre_content_path,
        out_dir=out_dir,
        amount=AMOUNT
    )
    
    # Process in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Use imap for better progress tracking
        results = list(tqdm(
            pool.imap(process_func, envelopes_metadata),
            total=len(envelopes_metadata),
            desc="Processing envelopes"
        ))
    
    # Flatten results
    meta = []
    for result in results:
        meta.extend(result)
    
    return meta

def process_timbre_pair(tc_pairs, envelopes_metadata, timbre_content_path, out_dir):
    """Original sequential processing function (kept for comparison)"""
    meta_chunk = []
    
    for env_data in tqdm(envelopes_metadata):
        # Random parameters in milliseconds
        ID = env_data["id"]
        STEM = env_data["stem"]
        A = env_data["attack"]
        D = env_data["decay"]
        H = env_data["hold"]
        S = env_data["sustain"]
        R = env_data["release"]
        length = env_data["length"]
        
        for amt in range(AMOUNT):
            tc_pair = random.choice(tc_pairs)
            audio, _ = sf.read(os.path.join(timbre_content_path, tc_pair))
            audio = audio.T
            audio = audio[:, START_POINT:]  # 3.5 secs
            note = np.tile(audio, 1)
        
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

            fname = f"ENV_{ID}_{tc_pair.split('.wav')[0]}_{STEM}_{amt}.wav"
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
    
    mp.set_start_method('spawn', force=True)
    
    timbre_content_path = f"{BASE_DIR}/rendered_one_shot" #"/mnt/gestalt/home/buffett/rendered_one_shot"
    timbre_content_pairs = []
    with open("stats/chosen_timbre_content_pairs.txt", "r") as f:
        for line in f:
            timbre_content_pairs.append(line.strip())
            
    with open(f"stats/envelopes_{SPLIT}.json", "r") as f:
        envelopes_metadata = json.load(f)
            
    print(f"Timbre-content pairs: {len(timbre_content_pairs)}")
    print(f"Envelopes metadata: {len(envelopes_metadata)}")

    # envelopes_metadata = envelopes_metadata[:10]
    
    # Use parallel processing
    meta = process_timbre_pair_parallel(timbre_content_pairs, envelopes_metadata, timbre_content_path, OUT_DIR)
    
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)