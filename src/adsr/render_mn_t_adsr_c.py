#!/usr/bin/env python
"""
Jun. 29, 2025
Generate an ADSR+Hold dataset from one-shot timbres and MIDI files.

Output naming pattern:  T###_ADSR###_C###.wav  (zero-padded indices)
"""

import os
import glob
import json
import multiprocessing as mp
import argparse
import time
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import soundfile as sf
import pretty_midi
from tqdm import tqdm
from util import SAMPLE_RATE, ms_to_samples
from librosa.effects import pitch_shift


# ---------------------------------------------------------------------
# Configuration – adjust to your folder layout
# ---------------------------------------------------------------------
BASE_DIR     = "/mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA"
SPLIT        = "evaluation" # train, evaluation
ADSR_PATH    = f"stats/envelopes_{SPLIT}_new.json"
TIMBRE_DIR   = f"{BASE_DIR}/rendered_one_shot_flat"   # folder with *.wav one-shots
MIDI_DIR     = f"../../info/{SPLIT}_midi_file_paths_satisfied.txt"    # folder with *.mid / *.midi files
OUTPUT_DIR   = f"{BASE_DIR}/rendered_mn_t_adsr_c/{SPLIT}"   # rendered dataset will be written here
MIDI_AMOUNT  = 100

TOTAL_DURATION = 3 # seconds
TRAINING_DURATION = 1 # seconds
REFERENCE_MIDI_NOTE = 48 # C3

# Multiprocessing configuration
NUM_PROCESSES = None  # Set to None for auto-detection, or specify a number (e.g., 4, 8, 16)
MAX_PROCESSES = 16    # Maximum number of processes to use

# Optimization configuration
USE_OPTIMIZED_RENDERING = True  # Use preloaded pitch-shifted timbres for better performance


# ---------------------------------------------------------------------
# 1.  Pitch-shifting function
# ---------------------------------------------------------------------   
def pitch_shift_audio(one_shot: np.ndarray, n_steps: float) -> np.ndarray:
    # C4 = MIDI note 60, so if note.pitch is 62 (D4), pitch_shift = 2 semitones up
    # If note.pitch is 58 (Bb3), pitch_shift = -2 semitones down
    return pitch_shift(one_shot, sr=SAMPLE_RATE, n_steps=n_steps)



# ---------------------------------------------------------------------
# 2.  ADSR-Envelope
# ---------------------------------------------------------------------
def process_adsr(adsr):
    A = adsr["attack"]
    D = adsr["decay"]
    H = adsr["hold"]
    S = adsr["sustain"]
    R = adsr["release"]

    A_samples = ms_to_samples(A)
    D_samples = ms_to_samples(D)
    H_samples = ms_to_samples(H)
    R_samples = ms_to_samples(R)
    length = A_samples + D_samples + H_samples + R_samples
    return length, A_samples, D_samples, S, R_samples


# ---------------------------------------------------------------------
# 4.  Optimized rendering with preloaded pitch-shifted timbres
# ---------------------------------------------------------------------
def render_midi_optimized(midi_path: str,
                         timbre: np.ndarray,
                         adsr_bank: List[Dict[str, float]]) -> List[np.ndarray]:
    """
    Optimized rendering that preloads pitch-shifted timbres for a MIDI file
    and applies all ADSR envelopes at once.
    
    Returns a list of audio buffers, one for each ADSR envelope.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Total length: last note-off + Release
    total_samples = int(np.ceil(TOTAL_DURATION * SAMPLE_RATE))
    
    # Use first instrument that has notes; otherwise skip
    instruments = [inst for inst in midi.instruments if inst.notes]
    if not instruments:
        # Return empty audio for all ADSR envelopes
        return [np.zeros(total_samples, dtype=np.float32) for _ in adsr_bank]

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
    for adsr in adsr_bank:
        mix = np.zeros(total_samples, dtype=np.float32)
        
        # Process ADSR data
        length, A_samples, D_samples, S, R_samples = process_adsr(adsr)
        
        # Process each note
        for note in inst.notes:
            if note.start >= TOTAL_DURATION:
                break
            
            start_samp = int(note.start * SAMPLE_RATE)
            note_dur_samp = int((note.end - note.start) * SAMPLE_RATE)
            note_dur_samp = max(note_dur_samp, 1)  # safety

            # Get preloaded pitch-shifted timbre
            seg = pitch_shifted_timbres[note.pitch]
            
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
            note_audio = seg[:actual_note_dur] * env[:actual_note_dur]

            # Mix into buffer (additive) - only the note part
            end_samp = min(start_samp + actual_note_dur, total_samples)
            if end_samp > start_samp:
                mix[start_samp:end_samp] += note_audio[:end_samp - start_samp]
            
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
                        mix[release_start_samp:release_end_samp] += release_audio[:release_end_samp - release_start_samp]

        # Simple peak-normalise to -1..1 (prevents clipping)
        max_amp = np.abs(mix).max() + 1e-9
        mix /= max_amp
        
        results.append(mix)
    
    return results


# ---------------------------------------------------------------------
# 3.  Metadata generation functions
# ---------------------------------------------------------------------
def get_timbre_info(timbre_path: str) -> Dict:
    return {
        "filename": os.path.basename(timbre_path)
    }

    
def get_midi_info(midi_path: str) -> Dict:
    midi = pretty_midi.PrettyMIDI(midi_path)
    return {
        "filename": os.path.basename(midi_path),
        "num_notes": len(midi.instruments[0].notes),
        "onset_seconds": [note.start for note in midi.instruments[0].notes if note.start <= TOTAL_DURATION - TRAINING_DURATION],
    }


# ---------------------------------------------------------------------
# 5.  Optimized worker function for multiprocessing
# ---------------------------------------------------------------------
def render_single_timbre_midi_combination(args: Tuple[int, int, np.ndarray, List[Dict], str, str]) -> List[Dict]:
    """
    Optimized worker function for multiprocessing.
    Renders all ADSR envelopes for a single timbre+MIDI combination.
    
    Args:
        args: Tuple of (t_idx, c_idx, timbre, adsr_bank, midi_path, output_dir)
    
    Returns:
        List of dictionaries with file metadata for all ADSR envelopes
    """
    t_idx, c_idx, timbre, adsr_bank, midi_path, output_dir = args
    
    try:
        # Render all ADSR envelopes for this timbre+MIDI combination
        audio_list = render_midi_optimized(midi_path, timbre, adsr_bank)
        
        results = []
        for a_idx, audio in enumerate(audio_list):
            # Generate output filename and path
            out_name = f"T{t_idx:03d}_ADSR{a_idx:03d}_C{c_idx:03d}.wav"
            out_path = os.path.join(output_dir, out_name)
            
            # Save the audio file
            sf.write(out_path, audio, SAMPLE_RATE)
            
            # Return metadata
            result = {
                "filename": out_name,
                "file_path": out_path,
                "timbre_index": t_idx,
                # "timbre_id": f"T{t_idx:03d}",
                "adsr_index": a_idx,
                # "adsr_id": f"ADSR{a_idx:03d}",
                "midi_index": c_idx,
                # "midi_id": f"C{c_idx:03d}",
                # "duration": float(len(audio) / SAMPLE_RATE),
                # "samples": len(audio),
                # "sample_rate": SAMPLE_RATE,
                # "channels": 1,
                # "peak_amplitude": float(np.abs(audio).max()),
                # "rms_amplitude": float(np.sqrt(np.mean(audio**2))),
                # "success": True
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        # Return error metadata for all ADSR envelopes
        results = []
        for a_idx in range(len(adsr_bank)):
            result = {
                "filename": f"T{t_idx:03d}_ADSR{a_idx:03d}_C{c_idx:03d}.wav",
                "timbre_index": t_idx,
                "adsr_index": a_idx,
                "midi_index": c_idx,
                "error": str(e),
                # "success": False
            }
            results.append(result)
        return results


# ---------------------------------------------------------------------
# 6.  Legacy worker function for backward compatibility
# ---------------------------------------------------------------------
def render_single_combination_legacy(args: Tuple[int, int, int, np.ndarray, Dict, str, str]) -> Dict:
    """
    Legacy worker function for multiprocessing.
    Renders a single combination of timbre, ADSR envelope, and MIDI file.
    
    Args:
        args: Tuple of (t_idx, a_idx, c_idx, timbre, adsr, midi_path, output_dir)
    
    Returns:
        Dictionary with file metadata
    """
    t_idx, a_idx, c_idx, timbre, adsr, midi_path, output_dir = args
    
    try:
        # Render the audio using the original function
        audio = render_midi_legacy(midi_path, timbre, adsr)
        
        # Generate output filename and path
        out_name = f"T{t_idx:03d}_ADSR{a_idx:03d}_C{c_idx:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        
        # Save the audio file
        sf.write(out_path, audio, SAMPLE_RATE)
        
        # Return metadata
        return {
            "filename": out_name,
            "file_path": out_path,
            "timbre_index": t_idx,
            "timbre_id": f"T{t_idx:03d}",
            "adsr_index": a_idx,
            "adsr_id": f"ADSR{a_idx:03d}",
            "midi_index": c_idx,
            "midi_id": f"C{c_idx:03d}",
            # "duration": float(len(audio) / SAMPLE_RATE),
            # "samples": len(audio),
            # "sample_rate": SAMPLE_RATE,
            # "channels": 1,
            # "peak_amplitude": float(np.abs(audio).max()),
            # "rms_amplitude": float(np.sqrt(np.mean(audio**2))),
            # "success": True
        }
    except Exception as e:
        # Return error metadata
        return {
            "filename": f"T{t_idx:03d}_ADSR{a_idx:03d}_C{c_idx:03d}.wav",
            "timbre_index": t_idx,
            "adsr_index": a_idx,
            "midi_index": c_idx,
            "error": str(e)
        }


# ---------------------------------------------------------------------
# 7.  Legacy rendering function (original implementation) Not used
# ---------------------------------------------------------------------
def render_midi_legacy(midi_path: str,
                      timbre: np.ndarray,
                      adsr: Dict[str, float]) -> np.ndarray:
    """
    Legacy rendering function (original implementation).
    Return a mono audio buffer containing the whole MIDI rendered with
    `timbre` as the sample and `adsr` as the envelope for every note.
    The timbre is assumed to be a C4 (MIDI note 60) one-shot sound.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Total length: last note-off + Release
    total_samples  = int(np.ceil(TOTAL_DURATION * SAMPLE_RATE))
    mix = np.zeros(total_samples, dtype=np.float32)

    # Use first instrument that has notes; otherwise skip
    instruments = [inst for inst in midi.instruments if inst.notes]
    if not instruments:
        return mix  # empty MIDI

    inst = instruments[0]
    
    # Notes pre-processing
    for note in inst.notes:
        if note.start >= TOTAL_DURATION:
            break
        
        start_samp = int(note.start * SAMPLE_RATE)
        note_dur_samp = int((note.end - note.start) * SAMPLE_RATE)
        note_dur_samp = max(note_dur_samp, 1)  # safety

        # Calculate pitch shift in semitones from C4 (MIDI note 60)
        pitch_shift = note.pitch - REFERENCE_MIDI_NOTE
        seg = pitch_shift_audio(timbre, pitch_shift) # 3 seconds
        
        # Process ADSR data
        length, A_samples, D_samples, S, R_samples = process_adsr(adsr)
        
        # Create envelope for the full note duration (including release)
        # The envelope should cover: note duration + release time
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
        # Ensure we don't exceed the length of the pitch-shifted audio
        actual_note_dur = min(note_dur_samp, len(seg))
        note_audio = seg[:actual_note_dur] * env[:actual_note_dur]

        # Mix into buffer (additive) - only the note part
        # Ensure we don't exceed the mix buffer bounds
        end_samp = min(start_samp + actual_note_dur, total_samples)
        if end_samp > start_samp:
            mix[start_samp:end_samp] += note_audio[:end_samp - start_samp]
        
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
                    mix[release_start_samp:release_end_samp] += release_audio[:release_end_samp - release_start_samp]

    # Simple peak-normalise to -1..1 (prevents clipping)
    max_amp = np.abs(mix).max() + 1e-9
    mix /= max_amp

    return mix


# ---------------------------------------------------------------------
# 5.  Main routine
# ---------------------------------------------------------------------
def main():
    start_time = time.time()
    
    timbre_paths = sorted(glob.glob(os.path.join(TIMBRE_DIR, "*.wav")))
    
    # Load all MIDI file paths
    with open(MIDI_DIR, "r") as f:
        midi_paths = [line.strip() for line in f.readlines()][:MIDI_AMOUNT]
        midi_paths = ["../" + midi_path for midi_path in midi_paths]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    

    # Pre-load timbres (converted to mono, SR-matched)
    timbres = []
    timbre_info = []
    for t in tqdm(timbre_paths, desc="Loading timbres"):
        x, sr = sf.read(t, always_2d=False)
        x = x.T
        if sr != SAMPLE_RATE:
            raise ValueError(f"{t} has SR={sr}; please resample to {SAMPLE_RATE} Hz.")
        timbres.append(x.astype(np.float32))
        timbre_info.append(get_timbre_info(t))



    # Generate ADSR set once
    with open(ADSR_PATH, "r") as f:
        adsr_bank: List[Dict[str, float]] = json.load(f)
    print(f"Render dataset with {len(timbre_paths)} timbres, {len(midi_paths)} MIDIs, {len(adsr_bank)} ADSR envelopes.")
    print(f"Using {'optimized' if USE_OPTIMIZED_RENDERING else 'legacy'} rendering approach")
    
    # Create metadata structure
    metadata = {
        "dataset_info": {
            "name": "T_ADSR_C_Dataset",
            "description": "Dataset generated from one-shot timbres with ADSR envelopes applied to MIDI files",
            "sample_rate": SAMPLE_RATE,
            "total_seconds": TOTAL_DURATION,
            "training_seconds": TRAINING_DURATION,
            "num_timbres": len(timbres),
            "num_adsr_envelopes": len(adsr_bank),
            "num_midi_files": len(midi_paths),
            "total_files": len(timbres) * len(adsr_bank) * len(midi_paths),
            "output_directory": OUTPUT_DIR,
            "split": SPLIT,
            "optimized_rendering": USE_OPTIMIZED_RENDERING
        },
        "adsr_envelopes": {},
        "timbres": {},
        "midi_files": {},
        "metadata": []
    }
    
    # Add ADSR envelope metadata
    for a_idx, adsr in enumerate(adsr_bank):
        metadata["adsr_envelopes"][f"ADSR{a_idx:03d}"] = {
            "attack": float(adsr["attack"]),
            "hold": float(adsr["hold"]),
            "decay": float(adsr["decay"]),
            "sustain": float(adsr["sustain"]),
            "release": float(adsr["release"]),
            "total_time": float(adsr["attack"] + adsr["hold"] + adsr["decay"] + adsr["release"])
        }
    
    # Add timbre metadata
    for t_idx, info in enumerate(timbre_info):
        metadata["timbres"][f"T{t_idx:03d}"] = info
    
    # Add MIDI file metadata
    for c_idx, midi_path in enumerate(midi_paths):
        midi_info = get_midi_info(midi_path)
        midi_info["filename"] = os.path.basename(midi_path)
        midi_info["full_path"] = midi_path
        metadata["midi_files"][f"C{c_idx:03d}"] = midi_info

    # Prepare all combinations for multiprocessing
    print("Preparing combinations for multiprocessing...")
    combinations = []
    
    if USE_OPTIMIZED_RENDERING:
        # Optimized approach: process all ADSR envelopes for each timbre+MIDI combination
        for t_idx, timbre in enumerate(timbres):
            for c_idx, midi_path in enumerate(midi_paths):
                combinations.append((t_idx, c_idx, timbre, adsr_bank, midi_path, OUTPUT_DIR))
    else:
        # Legacy approach: process each timbre+ADSR+MIDI combination separately
        for t_idx, timbre in enumerate(timbres):
            for a_idx, adsr in enumerate(adsr_bank):
                for c_idx, midi_path in enumerate(midi_paths):
                    combinations.append((t_idx, a_idx, c_idx, timbre, adsr, midi_path, OUTPUT_DIR))
    
    print(f"Total combinations to process: {len(combinations)}")
    
    # Use multiprocessing to render all combinations
    if NUM_PROCESSES is None:
        num_processes = min(mp.cpu_count(), MAX_PROCESSES)
    else:
        num_processes = min(NUM_PROCESSES, MAX_PROCESSES)
    
    print(f"Using {num_processes} processes for rendering...")
    
    # Process in chunks to avoid memory issues with large datasets
    chunk_size = 1000  # Process 1000 combinations at a time
    all_results = []
    
    render_start_time = time.time()
    
    # Add error handling for multiprocessing
    try:
        with mp.Pool(processes=num_processes) as pool:
            for i in range(0, len(combinations), chunk_size):
                chunk = combinations[i:i + chunk_size]
                print(f"Processing chunk {i//chunk_size + 1}/{(len(combinations) + chunk_size - 1)//chunk_size}")
                
                # Use tqdm to show progress for this chunk
                if USE_OPTIMIZED_RENDERING:
                    chunk_results = list(tqdm(
                        pool.imap(render_single_timbre_midi_combination, chunk),
                        total=len(chunk),
                        desc=f"Chunk {i//chunk_size + 1}"
                    ))
                else:
                    chunk_results = list(tqdm(
                        pool.imap(render_single_combination_legacy, chunk),
                        total=len(chunk),
                        desc=f"Chunk {i//chunk_size + 1}"
                    ))
                all_results.extend(chunk_results)
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to sequential processing...")
        all_results = []
        for combo in tqdm(combinations, desc="Rendering Dataset (Sequential)"):
            if USE_OPTIMIZED_RENDERING:
                all_results.append(render_single_timbre_midi_combination(combo))
            else:
                all_results.append(render_single_combination_legacy(combo))
    
    render_end_time = time.time()
    print(f"Rendering completed in {render_end_time - render_start_time:.2f} seconds")
    
    # Render all combinations
    if USE_OPTIMIZED_RENDERING:
        # Optimized approach: each result is a list of results for all ADSR envelopes
        for result_list in all_results:
            for result in result_list:  # Each result_list contains results for all ADSR envelopes
                metadata["metadata"].append(result)
    else:
        # Legacy approach: each result is a single result
        for result in all_results:
            metadata["metadata"].append(result)
    
    # Save metadata with each key to a separate JSON file
    for key, value in metadata.items():
        json_path = os.path.join(OUTPUT_DIR, f"{key}.json")
        with open(json_path, 'w') as f:
            json.dump(value, f, indent=2)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ADSR+Hold dataset from one-shot timbres and MIDI files")
    parser.add_argument("--split", type=str, default=SPLIT, help="Dataset split (default: train)")
    parser.add_argument("--adsr_path", type=str, default=ADSR_PATH, help="Path to ADSR envelope JSON file")
    parser.add_argument("--timbre_dir", type=str, default=TIMBRE_DIR, help="Directory containing one-shot timbres")
    parser.add_argument("--midi_dir", type=str, default=MIDI_DIR, help="Directory containing MIDI files")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory for the rendered dataset")
    parser.add_argument("--midi_amount", type=int, default=MIDI_AMOUNT, help="Number of MIDI files to process")
    parser.add_argument("--total_duration", type=float, default=TOTAL_DURATION, help="Total duration of the rendered audio")
    parser.add_argument("--reference_midi_note", type=int, default=REFERENCE_MIDI_NOTE, help="Reference MIDI note for pitch shifting")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES, help="Number of processes for multiprocessing")
    parser.add_argument("--max_processes", type=int, default=MAX_PROCESSES, help="Maximum number of processes to use")
    parser.add_argument("--use_optimized", action="store_true", default=USE_OPTIMIZED_RENDERING, help="Use optimized rendering with preloaded pitch-shifted timbres")
    parser.add_argument("--use_legacy", action="store_true", help="Use legacy rendering (slower but uses less memory)")
    args = parser.parse_args()

    SPLIT        = args.split
    ADSR_PATH    = args.adsr_path
    TIMBRE_DIR   = args.timbre_dir
    MIDI_DIR     = args.midi_dir
    OUTPUT_DIR   = args.output_dir
    MIDI_AMOUNT  = args.midi_amount
    TOTAL_DURATION = args.total_duration
    REFERENCE_MIDI_NOTE = args.reference_midi_note
    NUM_PROCESSES = args.num_processes
    MAX_PROCESSES = args.max_processes
    USE_OPTIMIZED_RENDERING = args.use_optimized and not args.use_legacy

    print(args)
    main()
