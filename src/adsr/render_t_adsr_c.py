#!/usr/bin/env python
"""
Jun. 29, 2025
Generate an ADSR+Hold dataset from one-shot timbres and MIDI files.

Output naming pattern:  T###_ADSR###_C###.wav  (zero-padded indices)
"""

import os
import random
import glob
import json
import librosa
from typing import Dict, List

import numpy as np
import soundfile as sf
import pretty_midi
from tqdm import tqdm
from util import SAMPLE_RATE, ms_to_samples
from scipy import signal
from librosa.effects import pitch_shift, time_stretch


# ---------------------------------------------------------------------
# Configuration – adjust to your folder layout
# ---------------------------------------------------------------------
SPLIT        = "train"
ADSR_PATH    = "stats/envelopes_train_new.json"
TIMBRE_DIR   = "../../rendered_one_shot"   # folder with *.wav one-shots
MIDI_DIR     = f"../../info/{SPLIT}_midi_file_paths_satisfied.txt"    # folder with *.mid / *.midi files
OUTPUT_DIR   = "../../rendered_t_adsr_c"   # rendered dataset will be written here
START_POINT  = 44100 * 1
END_POINT    = 44100 * 4
MIDI_AMOUNT  = 200 #500

TOTAL_DURATION = 3 # seconds
REFERENCE_MIDI_NOTE = 60 # C4


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
# 3.  Render a single MIDI with one timbre + one ADSR
# ---------------------------------------------------------------------
def render_midi(midi_path: str,
                timbre: np.ndarray,
                adsr: Dict[str, float]) -> np.ndarray:
    """
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
        note_audio = seg[:note_dur_samp] * env[:note_dur_samp]

        # Mix into buffer (additive) - only the note part
        mix[start_samp:start_samp + note_dur_samp] += note_audio
        
        # Handle release phase separately - it continues after note ends
        if R_samples > 0 and note_dur_samp < total_env_samples:
            release_start = note_dur_samp
            release_end = min(release_start + R_samples, total_env_samples)
            if release_end > release_start:
                # For release, we need to continue the timbre (could be silence or last sample)
                release_timbre = np.full(release_end - release_start, seg[-1] if len(seg) > 0 else 0)
                release_env = env[release_start:release_end]
                release_audio = release_timbre * release_env
                
                # Add release to the mix buffer
                release_start_samp = start_samp + note_dur_samp
                release_end_samp = min(release_start_samp + len(release_audio), total_samples)
                if release_end_samp > release_start_samp:
                    mix[release_start_samp:release_end_samp] += release_audio[:release_end_samp - release_start_samp]

    # Simple peak-normalise to -1..1 (prevents clipping)
    max_amp = np.abs(mix).max() + 1e-9
    mix /= max_amp

    return mix


# ---------------------------------------------------------------------
# 3.  Metadata generation functions
# ---------------------------------------------------------------------
def get_timbre_info(timbre_path: str) -> Dict:
    """Extract metadata from timbre file."""
    try:
        x, sr = sf.read(timbre_path, always_2d=False)
        return {
            "filename": os.path.basename(timbre_path),
            "sample_rate": int(sr),
            "duration": float(len(x) / sr),
            "channels": x.ndim,
            "samples": len(x),
            "dtype": str(x.dtype)
        }
    except Exception as e:
        return {
            "filename": os.path.basename(timbre_path),
            "error": str(e)
        }
    
    
def get_midi_info(midi_path: str) -> Dict:
    """Extract metadata from MIDI file."""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        return {
            "filename": os.path.basename(midi_path),
            "duration": float(midi.get_end_time()),
            "num_notes": len(midi.instruments[0].notes),
            "onset_seconds": [note.start for note in midi.instruments[0].notes if note.start < TOTAL_DURATION],
        }
    except Exception as e:
        return {
            "filename": os.path.basename(midi_path),
            "error": str(e)
        }


# ---------------------------------------------------------------------
# 4.  Main routine
# ---------------------------------------------------------------------
def main():
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
        x = x[:, START_POINT:END_POINT]  # 3 secs
        if x.ndim > 1:
            x = x.mean(axis=0)  # stereo → mono
        if sr != SAMPLE_RATE:
            raise ValueError(f"{t} has SR={sr}; please resample to {SAMPLE_RATE} Hz.")
        timbres.append(x.astype(np.float32))
        timbre_info.append(get_timbre_info(t))



    # Generate ADSR set once
    with open(ADSR_PATH, "r") as f:
        adsr_bank: List[Dict[str, float]] = json.load(f)
    print(f"Render dataset with {len(timbre_paths)} timbres, {len(midi_paths)} MIDIs, {len(adsr_bank)} ADSR envelopes.")
    
    # Create metadata structure
    metadata = {
        "dataset_info": {
            "name": "T_ADSR_C_Dataset",
            "description": "Dataset generated from one-shot timbres with ADSR envelopes applied to MIDI files",
            "sample_rate": SAMPLE_RATE,
            "start_point": START_POINT,
            "end_point": END_POINT,
            "num_timbres": len(timbres),
            "num_adsr_envelopes": len(adsr_bank),
            "num_midi_files": len(midi_paths),
            "total_files": len(timbres) * len(adsr_bank) * len(midi_paths),
            "output_directory": OUTPUT_DIR,
            "split": SPLIT
        },
        "adsr_envelopes": {},
        "timbres": {},
        "midi_files": {},
        "generated_files": []
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

    # Rendering loop
    for t_idx, timbre in enumerate(tqdm(timbres, desc="Rendering Dataset")):
        for a_idx, adsr in enumerate(adsr_bank):
            for c_idx, midi_path in enumerate(midi_paths):
                audio = render_midi(midi_path, timbre, adsr)

                out_name = f"T{t_idx:03d}_ADSR{a_idx:03d}_C{c_idx:03d}.wav"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                sf.write(out_path, audio, SAMPLE_RATE)
                
                # Add generated file metadata
                file_metadata = {
                    "filename": out_name,
                    "file_path": out_path,
                    "timbre_index": t_idx,
                    "timbre_id": f"T{t_idx:03d}",
                    "adsr_index": a_idx,
                    "adsr_id": f"ADSR{a_idx:03d}",
                    "midi_index": c_idx,
                    "midi_id": f"C{c_idx:03d}",
                    "duration": float(len(audio) / SAMPLE_RATE),
                    "samples": len(audio),
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "peak_amplitude": float(np.abs(audio).max()),
                    "rms_amplitude": float(np.sqrt(np.mean(audio**2)))
                }
                metadata["generated_files"].append(file_metadata)
                break
            break
        break

    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(metadata['generated_files'])} files")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()