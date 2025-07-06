#!/usr/bin/env python
"""
Simplified one-shot sample renderer with ADSR envelopes and pitch shifting.
Generates audio files with naming pattern: T###_ADSR###_C###.wav
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
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = "/home/buffett/nas_data/EDM_FAC_NEW_DATA" # "/Users/buffettliu/Desktop/Music_AI/Codes/serum_render"
SPLIT = "train"
ADSR_PATH = "stats/envelopes_train_new.json"
TIMBRE_DIR = f"{BASE_DIR}/rendered_one_shot_flat"
OUTPUT_DIR = f"{BASE_DIR}/rendered_ss_t_adsr_c"
START_POINT = 44100 * 0
END_POINT = 44100 * 1

TOTAL_DURATION = 1  # seconds
REFERENCE_MIDI_NOTE = 48  # C3

# Multiprocessing
NUM_PROCESSES = None
MAX_PROCESSES = 16

# C_NOTES mapping for pitch shifting
BASE_NOTES = {'2': 36, '3': 48, '4': 60}
CLICK = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
C_NOTES = {f'{note}{octave}': CLICK[note] + BASE_NOTES[octave] 
           for note in CLICK.keys() for octave in BASE_NOTES.keys()}

# Create indexed C_NOTES with 0-based indexing starting from smallest MIDI note
C_NOTES_INDEXED = {}
sorted_notes = sorted(C_NOTES.items(), key=lambda x: x[1])  # Sort by MIDI note value
for idx, (note_name, midi_note) in enumerate(sorted_notes):
    C_NOTES_INDEXED[idx] = note_name


# ---------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------
def pitch_shift_audio(one_shot: np.ndarray, n_steps: float) -> np.ndarray:
    """Pitch shift audio by n_steps semitones."""
    return pitch_shift(one_shot, sr=SAMPLE_RATE, n_steps=n_steps)


def create_adsr_envelope(total_samples: int, attack_ms: float, decay_ms: float, 
                        sustain_level: float, release_ms: float) -> np.ndarray:
    """Create ADSR envelope."""
    attack_samples = ms_to_samples(attack_ms)
    decay_samples = ms_to_samples(decay_ms)
    release_samples = ms_to_samples(release_ms)
    
    env = np.zeros(total_samples, dtype=np.float32)
    
    # Attack phase (0 to 1)
    if attack_samples > 0:
        attack_end = min(attack_samples, total_samples)
        env[:attack_end] = np.linspace(0, 1, attack_end, endpoint=False)
    
    # Decay phase (1 to sustain level)
    if decay_samples > 0 and attack_samples < total_samples:
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, total_samples)
        if decay_end > decay_start:
            env[decay_start:decay_end] = np.linspace(1, sustain_level, decay_end - decay_start, endpoint=False)
    
    # Sustain phase (sustain level)
    sustain_start = min(attack_samples + decay_samples, total_samples)
    sustain_end = min(sustain_start + (total_samples - sustain_start - release_samples), total_samples)
    if sustain_end > sustain_start:
        env[sustain_start:sustain_end] = sustain_level
    
    # Release phase (sustain to 0)
    if release_samples > 0 and sustain_end < total_samples:
        release_start = sustain_end
        release_end = min(release_start + release_samples, total_samples)
        if release_end > release_start:
            env[release_start:release_end] = np.linspace(sustain_level, 0, release_end - release_start, endpoint=False)
    
    return env


def render_one_shot_with_adsr(one_shot: np.ndarray, note_name: str, adsr_params: Dict) -> np.ndarray:
    """
    Render one-shot sample with pitch shifting and ADSR envelope.
    
    Args:
        one_shot: Original one-shot audio
        note_name: Note name (e.g., 'C3', 'D4')
        adsr_params: ADSR parameters dictionary
    
    Returns:
        Rendered audio with pitch shift and ADSR envelope applied
    """
    # Get target MIDI note from C_NOTES
    if note_name not in C_NOTES:
        raise ValueError(f"Note {note_name} not found in C_NOTES mapping")
    
    target_midi_note = C_NOTES[note_name]
    pitch_shift_steps = target_midi_note - REFERENCE_MIDI_NOTE
    
    # Apply pitch shift
    pitch_shifted = pitch_shift_audio(one_shot, pitch_shift_steps)
    
    # Create ADSR envelope
    total_samples = len(pitch_shifted)
    envelope = create_adsr_envelope(
        total_samples,
        adsr_params["attack"],
        adsr_params["decay"], 
        adsr_params["sustain"],
        adsr_params["release"]
    )
    
    # Apply envelope to pitch-shifted audio
    rendered = pitch_shifted * envelope
    
    # Normalize to prevent clipping
    max_amp = np.abs(rendered).max() + 1e-9
    rendered /= max_amp
    
    return rendered


def render_single_combination(args: Tuple[int, int, int, np.ndarray, Dict, str, str]) -> Dict:
    """
    Render a single combination of timbre, ADSR, and note.
    
    Args:
        args: (timbre_idx, adsr_idx, note_idx, timbre, adsr_params, note_name, output_dir)
    
    Returns:
        Metadata dictionary
    """
    timbre_idx, adsr_idx, note_idx, timbre, adsr_params, note_name, output_dir = args
    
    try:
        # Render the audio
        audio = render_one_shot_with_adsr(timbre, note_name, adsr_params)
        
        # Generate output filename
        out_name = f"T{timbre_idx:03d}_ADSR{adsr_idx:03d}_C{note_idx:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        
        # Save audio file
        sf.write(out_path, audio, SAMPLE_RATE)
        
        return {
            "filename": out_name,
            "file_path": out_path,
            "timbre_index": timbre_idx,
            "adsr_index": adsr_idx,
            "note_index": note_idx,
            "note_name": note_name,
            "duration": float(len(audio) / SAMPLE_RATE),
            "samples": len(audio),
            "sample_rate": SAMPLE_RATE,
            "peak_amplitude": float(np.abs(audio).max()),
            "rms_amplitude": float(np.sqrt(np.mean(audio**2))),
            "success": True
        }
        
    except Exception as e:
        return {
            "filename": f"T{timbre_idx:03d}_ADSR{adsr_idx:03d}_C{note_idx:03d}.wav",
            "timbre_index": timbre_idx,
            "adsr_index": adsr_idx,
            "note_index": note_idx,
            "error": str(e),
            "success": False
        }


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def main():
    start_time = time.time()
    
    # Load one-shot timbres
    timbre_paths = sorted(glob.glob(os.path.join(TIMBRE_DIR, "*.wav")))
    print(f"Found {len(timbre_paths)} timbre files")
    
    # Load ADSR parameters
    with open(ADSR_PATH, "r") as f:
        adsr_bank = json.load(f)
    print(f"Loaded {len(adsr_bank)} ADSR envelopes")
    
    # Get note names from C_NOTES_INDEXED (sorted by MIDI note value)
    note_names = list(C_NOTES_INDEXED.values())
    print(f"Using {len(note_names)} notes: {note_names}")
    print(f"Note indices: {list(C_NOTES_INDEXED.keys())}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load timbres
    timbres = []
    for path in tqdm(timbre_paths, desc="Loading timbres"):
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # Convert to mono
        audio = audio[START_POINT:END_POINT]  # Extract segment
        if sr != SAMPLE_RATE:
            raise ValueError(f"Sample rate mismatch: {sr} != {SAMPLE_RATE}")
        timbres.append(audio.astype(np.float32))
    
    # Prepare all combinations
    combinations = []
    for t_idx, timbre in enumerate(timbres):
        for a_idx, adsr_params in enumerate(adsr_bank):
            for n_idx, note_name in enumerate(note_names):
                combinations.append((t_idx, a_idx, n_idx, timbre, adsr_params, note_name, OUTPUT_DIR))
    
    print(f"Total combinations: {len(combinations)}")
    
    # Setup multiprocessing
    num_processes = min(NUM_PROCESSES or mp.cpu_count(), MAX_PROCESSES)
    print(f"Using {num_processes} processes")
    
    # Process combinations
    successful_files = []
    failed_files = []
    
    try:
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(render_single_combination, combinations),
                total=len(combinations),
                desc="Rendering combinations"
            ))
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to sequential processing...")
        results = []
        for combo in tqdm(combinations, desc="Rendering (sequential)"):
            results.append(render_single_combination(combo))
    
    # Collect results
    for result in results:
        if result.get("success", False):
            successful_files.append(result)
        else:
            failed_files.append(result)
            print(f"Failed: {result['filename']} - {result.get('error', 'Unknown error')}")
    
    print(f"Successfully generated: {len(successful_files)} files")
    if failed_files:
        print(f"Failed: {len(failed_files)} files")
    
    # Save metadata
    metadata = {
        "dataset_info": {
            "num_timbres": len(timbres),
            "num_adsr_envelopes": len(adsr_bank),
            "num_notes": len(note_names),
            "total_files": len(combinations),
            "successful_files": len(successful_files),
            "failed_files": len(failed_files),
            "sample_rate": SAMPLE_RATE,
            "reference_midi_note": REFERENCE_MIDI_NOTE,
            "c_notes_mapping": C_NOTES,
            "c_notes_indexed": C_NOTES_INDEXED
        },
        "files": successful_files
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render one-shot samples with ADSR envelopes")
    parser.add_argument("--timbre_dir", type=str, default=TIMBRE_DIR, help="Directory with one-shot timbres")
    parser.add_argument("--adsr_path", type=str, default=ADSR_PATH, help="Path to ADSR parameters JSON")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--start_point", type=int, default=START_POINT, help="Start point in samples")
    parser.add_argument("--end_point", type=int, default=END_POINT, help="End point in samples")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES, help="Number of processes")
    args = parser.parse_args()
    
    TIMBRE_DIR = args.timbre_dir
    ADSR_PATH = args.adsr_path
    OUTPUT_DIR = args.output_dir
    START_POINT = args.start_point
    END_POINT = args.end_point
    NUM_PROCESSES = args.num_processes
    
    main()
