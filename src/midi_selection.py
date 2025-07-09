import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from tqdm import tqdm

# Suppress pretty_midi warnings about invalid MIDI file types
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")

UNIQUE_NOTES_THRESHOLD = 5 # 5
DURATION_SECOND = 3 # 10
SPLIT = ["train", "evaluation"]
AMOUNT = [200, 50]

def plot_midi_piano_roll(midi_path, split, max_duration=DURATION_SECOND):
    satisfied = True
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    fs = 100  # Time resolution in frames per second
    piano_roll = midi_data.get_piano_roll(fs=fs)

    end_frame = min(piano_roll.shape[1], int(max_duration * fs))
    midi_name = midi_path.split('/')[-1].split('.mid')[0]

    # ----- Check 1: <5 unique notes -----
    active_notes, _ = np.nonzero(piano_roll[:, :end_frame])
    unique_notes = np.unique(active_notes)
    if len(unique_notes) < UNIQUE_NOTES_THRESHOLD:
        # print(f"WARNING: {midi_name} has only {len(unique_notes)} unique MIDI notes")
        satisfied = False
        
    # ----- Check 2: bpm == 120 ----- 
    tempo_times, tempi = midi_data.get_tempo_changes()
    if tempi.size > 0:
        for t, bpm in zip(tempo_times, tempi):
            if bpm != 120:
                satisfied = False
                break
    
        
    # ----- Check 3: >0.5s gap with no note -----
    active_per_frame = (piano_roll[:, :end_frame] > 0).sum(axis=0)
    silent_frames = (active_per_frame == 0).astype(int)

    # Find longest silent streak
    silent_streaks = []
    current_streak = 0
    for val in silent_frames:
        if val == 1:
            current_streak += 1
        else:
            if current_streak > 0:
                silent_streaks.append(current_streak)
                current_streak = 0
    if current_streak > 0:
        silent_streaks.append(current_streak)

    if any(streak > fs * 0.5 for streak in silent_streaks):  # > 0.5 sec gap
        # print(f"WARNING: {midi_name} has silence longer than 0.5s")
        satisfied = False
        
    if satisfied:
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
        
        plt.savefig(f"../vis_results/{split}/{midi_name}.png")
        plt.close()
    
    return satisfied
  
        
    
if __name__ == "__main__":
    for split, amount in zip(SPLIT, AMOUNT):
        counter = 0
        satisfied_midi_file_paths = []
        MIDI_DIR = f"../midi/midi_files/{split}/midi"
        midi_file_paths = [os.path.join(MIDI_DIR, f) 
                        for f in os.listdir(MIDI_DIR) if f.endswith('.mid')]

        os.makedirs("../vis_results", exist_ok=True)
        os.makedirs(f"../vis_results/{split}", exist_ok=True)
        
        for mfp in tqdm(midi_file_paths):
            if plot_midi_piano_roll(mfp, split):
                satisfied_midi_file_paths.append(mfp)

                counter += 1
                if counter == amount:
                    break
                
        with open(f"../info/{split}_midi_file_paths_satisfied.txt", "w") as f:
            for mfp in satisfied_midi_file_paths:
                f.write(mfp + "\n")
                
        print(f"Saved {len(satisfied_midi_file_paths)} satisfied MIDI files for {split}")
