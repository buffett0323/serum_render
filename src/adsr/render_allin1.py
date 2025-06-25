import numpy as np, soundfile as sf, json, random, os
import nnAudio.features
import torch
from pathlib import Path
from pedalboard import Pedalboard, Gain
from tqdm import tqdm
from util import SAMPLE_RATE, ms_to_samples, samples_to_ms, SERUM_ADSR
import multiprocessing as mp
from functools import partial
from collections import defaultdict
import pickle
import psutil

SAMPLE_RATE = 44100
START_POINT = 22050
AMOUNT = 10
UNIT_SECONDS = 2.97
UNIT_LENGTH = int(UNIT_SECONDS * SAMPLE_RATE)
SPLIT = "test" # val # test
STEMS = ["lead", "keys", "pad", "pluck", "synth", "vox"]
BASE_DIR = "/mnt/gestalt/home/buffett" #"/home/buffett/dataset"
OUT_DIR = Path(f"{BASE_DIR}/rendered_adsr_unpaired_mel_npy/{SPLIT}") #"/mnt/gestalt/home/buffett/rendered_adsr_dataset"
OUT_DIR.mkdir(exist_ok=True)

# Memory management settings
CHUNK_SIZE = 50  # Process 50 envelopes at a time (adjust based on available memory)
MEL_BATCH_SIZE = 32  # GPU batch size for mel-spectrogram computation

# Global cache for audio files to avoid repeated disk I/O
AUDIO_CACHE = {}

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


def load_audio_cached(timbre_content_path, tc_pair):
    """Load audio with caching to avoid repeated disk I/O"""
    if tc_pair not in AUDIO_CACHE:
        audio, _ = sf.read(os.path.join(timbre_content_path, tc_pair))
        audio = audio.T
        audio = audio[:, START_POINT:]  # 3.5 secs
        AUDIO_CACHE[tc_pair] = audio
    return AUDIO_CACHE[tc_pair]


def process_single_envelope(env_data, tc_pairs, timbre_content_path, out_dir, amount):
    """Process a single envelope with multiple timbre content pairs - CPU only, no mel computation"""
    meta_chunk = []
    audio_data_chunk = []
    
    # Random parameters in milliseconds
    ID = env_data["id"]
    STEM = env_data["stem"]
    A = env_data["attack"]
    D = env_data["decay"]
    H = env_data["hold"]
    S = env_data["sustain"]
    R = env_data["release"]
    length = env_data["length"]
    
    # Pre-compute ADSR parameters once
    A_samples = ms_to_samples(A)
    D_samples = ms_to_samples(D)
    H_samples = ms_to_samples(H)
    R_samples = ms_to_samples(R)
    total_length = A_samples + D_samples + H_samples + R_samples
    
    # Pre-compute envelope once
    env = adsr_env(total_length, A_samples, D_samples, S, R_samples)
    env = env.reshape(1, -1)  # Reshape to (1, length) for broadcasting
    
    # Create Pedalboard once
    board = Pedalboard([Gain(gain_db=0.0)])
    
    for amt in range(amount):
        try:
            tc_pair = random.choice(tc_pairs)
            audio = load_audio_cached(timbre_content_path, tc_pair)
            
            # Use the pre-computed envelope
            signal = audio[:, :total_length] * env

            # Pedalboard mixing
            wav = board(signal, SAMPLE_RATE) 
            assert wav.shape[0] == 2, "Pedalboard should output stereo audio"
            
            # Convert to mono using numpy
            wav = wav.mean(axis=0, keepdims=True)  # Convert to mono
            
            current_length = wav.shape[1]
            if current_length < UNIT_LENGTH:
                # Pad with zeros
                padding = UNIT_LENGTH - current_length
                wav = np.pad(wav, ((0, 0), (0, padding)), mode='constant')
            elif current_length > UNIT_LENGTH:
                # Truncate
                wav = wav[:, :UNIT_LENGTH]
            
            fname = f"ENV_{ID}_{tc_pair.split('.wav')[0]}_{STEM}_{amt}_mel.npy"
            
            # Store audio data and metadata for later GPU processing
            audio_data_chunk.append({
                'audio': wav,
                'fname': fname,
                'out_dir': str(out_dir)
            })

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
    
    return meta_chunk, audio_data_chunk


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def estimate_memory_per_audio():
    """Estimate memory usage per audio file in MB"""
    # UNIT_LENGTH samples * 4 bytes (float32) * 1 channel (mono)
    return (UNIT_LENGTH * 4) / 1024 / 1024

def process_mel_spectrograms_gpu(audio_data_batch, mel_converter, device, batch_size=MEL_BATCH_SIZE):
    """Process mel-spectrograms in batches on GPU"""
    results = []
    
    # Process in batches for better GPU utilization
    for i in range(0, len(audio_data_batch), batch_size):
        batch = audio_data_batch[i:i + batch_size]
        
        # Prepare batch
        audio_tensors = []
        filenames = []
        out_dirs = []
        
        for item in batch:
            audio_tensors.append(torch.from_numpy(item['audio']).float())
            filenames.append(item['fname'])
            out_dirs.append(item['out_dir'])
        
        # Stack and move to GPU
        audio_batch_tensor = torch.stack(audio_tensors, dim=0).to(device)
        
        # Compute mel-spectrograms
        with torch.no_grad():
            mel_batch = mel_converter(audio_batch_tensor)  # [batch_size, n_mels, time]
        
        # Save results
        for mel, fname, out_dir in zip(mel_batch, filenames, out_dirs):
            np.save(Path(out_dir) / fname, mel.cpu().numpy().astype(np.float32))
            results.append(fname)
    
    return results

def process_timbre_pair_parallel(tc_pairs, envelopes_metadata, timbre_content_path, 
                                 out_dir, mel_converter, device, num_processes=None, chunk_size=CHUNK_SIZE):
    """Process envelopes in parallel using multiprocessing, then GPU mel computation in chunks"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} processes for audio processing")
    print(f"Processing in chunks of {chunk_size} envelopes to manage memory")
    
    # Memory estimation
    mem_per_audio = estimate_memory_per_audio()
    max_audio_per_chunk = chunk_size * AMOUNT
    estimated_chunk_memory = max_audio_per_chunk * mem_per_audio
    print(f"Estimated memory per chunk: {estimated_chunk_memory:.2f} MB")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_single_envelope,
        tc_pairs=tc_pairs,
        timbre_content_path=timbre_content_path,
        out_dir=out_dir,
        amount=AMOUNT
    )
    
    # Process envelopes in chunks to manage memory
    all_meta = []
    total_envelopes = len(envelopes_metadata)
    
    for chunk_start in range(0, total_envelopes, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_envelopes)
        chunk_envelopes = envelopes_metadata[chunk_start:chunk_end]
        
        print(f"\nProcessing chunk {chunk_start//chunk_size + 1}/{(total_envelopes + chunk_size - 1)//chunk_size} "
              f"(envelopes {chunk_start}-{chunk_end-1})")
        print(f"Memory before chunk: {get_memory_usage():.2f} GB")
        
        # Process audio in parallel on CPU for this chunk
        with mp.Pool(processes=num_processes) as pool:
            # Use imap for better progress tracking
            results = list(tqdm(
                pool.imap(process_func, chunk_envelopes),
                total=len(chunk_envelopes),
                desc=f"Processing audio chunk {chunk_start//chunk_size + 1}"
            ))
        
        # Separate metadata and audio data for this chunk
        chunk_meta = []
        chunk_audio_data = []
        for result in results:
            meta_chunk, audio_data_chunk = result
            chunk_meta.extend(meta_chunk)
            chunk_audio_data.extend(audio_data_chunk)
        
        print(f"Processing {len(chunk_audio_data)} mel-spectrograms on GPU for this chunk...")
        print(f"Memory after audio processing: {get_memory_usage():.2f} GB")
        
        # Process mel-spectrograms for this chunk on GPU
        process_mel_spectrograms_gpu(chunk_audio_data, mel_converter, device)
        
        # Add chunk metadata to total
        all_meta.extend(chunk_meta)
        
        # Clear chunk data to free memory
        del chunk_audio_data
        del chunk_meta
        
        print(f"Completed chunk {chunk_start//chunk_size + 1}")
        print(f"Memory after cleanup: {get_memory_usage():.2f} GB")
    
    return all_meta

def get_optimal_chunk_size(total_envelopes, amount_per_envelope, target_memory_gb=2.0):
    """Calculate optimal chunk size based on available memory"""
    mem_per_audio = estimate_memory_per_audio()
    total_memory_mb = target_memory_gb * 1024
    
    # Calculate how many audio files we can fit in target memory
    max_audio_files = int(total_memory_mb / mem_per_audio)
    
    # Calculate chunk size (envelopes per chunk)
    optimal_chunk_size = max(1, max_audio_files // amount_per_envelope)
    
    # Don't make chunks too large
    optimal_chunk_size = min(optimal_chunk_size, 200)
    
    return optimal_chunk_size

if __name__ == "__main__":
    
    mp.set_start_method('spawn', force=True)
    
    timbre_content_path = f"{BASE_DIR}/rendered_one_shot"
    timbre_content_pairs = []
    with open("stats/chosen_timbre_content_pairs.txt", "r") as f:
        for line in f:
            timbre_content_pairs.append(line.strip())
            
    with open(f"stats/envelopes_{SPLIT}.json", "r") as f:
        envelopes_metadata = json.load(f)
    
    # envelopes_metadata = envelopes_metadata[:1000]
    
    # Calculate optimal chunk size based on available memory
    total_envelopes = len(envelopes_metadata)
    optimal_chunk_size = get_optimal_chunk_size(total_envelopes, AMOUNT, target_memory_gb=2.0)
    print(f"Total envelopes: {total_envelopes}")
    print(f"Optimal chunk size: {optimal_chunk_size} (will use ~{optimal_chunk_size * AMOUNT * estimate_memory_per_audio():.1f} MB per chunk)")
    
    # Pre-load all audio files to avoid disk I/O during processing
    print("Pre-loading audio files...")
    for tc_pair in tqdm(timbre_content_pairs, desc="Loading audio files"):
        if tc_pair not in AUDIO_CACHE:
            audio, _ = sf.read(os.path.join(timbre_content_path, tc_pair))
            audio = audio.T
            audio = audio[:, START_POINT:]  # 3.5 secs
            AUDIO_CACHE[tc_pair] = audio
    print(f"Loaded {len(AUDIO_CACHE)} audio files into memory")
    print(f"Memory after loading audio cache: {get_memory_usage():.2f} GB")
            
    print(f"Timbre-content pairs: {len(timbre_content_pairs)}")
    print(f"Envelopes metadata: {len(envelopes_metadata)}")

    # Use GPU for mel-spectrogram computation
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mel_converter = nnAudio.features.MelSpectrogram(
        sr=SAMPLE_RATE,
        n_mels=128,
        fmin=20,
        fmax=22050,
        hop_length=512,
        n_fft=2048,
        window='hann',
        center=True,
        power=2.0,
    ).to(device)
    
    # Use parallel processing for audio, then GPU for mel-spectrograms
    meta = process_timbre_pair_parallel(timbre_content_pairs, envelopes_metadata, timbre_content_path, 
                                        OUT_DIR, mel_converter, device, chunk_size=optimal_chunk_size)
    
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    