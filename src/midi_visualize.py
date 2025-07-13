import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

DURATION_SECOND = 1 # 10
SPLIT = ["train", "evaluation"]
AMOUNT = [100, 10] # [100, 10]
OUTPUT_DIR = "../midi_vis_1secs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_midi_piano_roll(midi_path, split, max_duration=DURATION_SECOND):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    fs = 1000  # Time resolution in frames per second
    piano_roll = midi_data.get_piano_roll(fs=fs)

    end_frame = min(piano_roll.shape[1], int(max_duration * fs))
    midi_name = midi_path.split('/')[-1].split('.mid')[0]
    
    # Visualization part
    plt.figure(figsize=(14, 6))
    plt.imshow(piano_roll[:, :end_frame], aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel("Time (frames)")
    plt.ylabel("MIDI note number")
    plt.title(f"Piano Roll of: {midi_name}")
    plt.colorbar(label='Velocity')
    plt.grid(True)
    
    x_ticks = np.arange(0, end_frame, fs)
    x_labels = [f"{t/fs:.1f}" for t in x_ticks]
    plt.xticks(x_ticks, x_labels)
    
    # Add note annotations and onset markers
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Only process notes within the time window
            if note.start < max_duration:
                # Convert time to frame index
                start_frame = int(note.start * fs)
                end_frame_note = int(note.end * fs)
                
                # Only annotate if note is within the displayed range
                if start_frame < end_frame:
                    # Get note name
                    note_name = pretty_midi.note_number_to_name(note.pitch)
                    
                    # Mark onset with a vertical line
                    plt.axvline(x=start_frame, color='red', alpha=0.7, linestyle='--', linewidth=1)
                    
                    # Add note annotation at the onset
                    plt.annotate(f'{note_name}\n{note.start:.2f}s', 
                               xy=(start_frame, note.pitch), 
                               xytext=(start_frame + 50, note.pitch + 2),
                               fontsize=8, color='white', weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8),
                               arrowprops=dict(arrowstyle="->", color='red', alpha=0.7))
    
    plt.savefig(f"{OUTPUT_DIR}/{split}/{midi_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
  
        
    
if __name__ == "__main__":
    for split, amount in zip(SPLIT, AMOUNT):
        os.makedirs(f"{OUTPUT_DIR}/{split}", exist_ok=True)
        with open(f"../info/{split}_midi_file_paths_satisfied.txt", "r") as f:
            midi_file_paths = f.readlines()
            midi_file_paths = [mfp.strip() for mfp in midi_file_paths]
            midi_file_paths = midi_file_paths[:amount]
            
        for mfp in tqdm(midi_file_paths):
            plot_midi_piano_roll(mfp, split)
