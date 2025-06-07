import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

UNIQUE_NOTES_THRESHOLD = 3 # 5
DURATION_SECOND = 5 # 10

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
    
        
    # ----- Check 3: >1s gap with no note -----
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

    if any(streak > fs for streak in silent_streaks):  # > 1 sec gap
        # print(f"WARNING: {midi_name} has silence longer than 1s")
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
        
        plt.savefig(f"../vis_results/{split}/{midi_name}.png")
        plt.close()
    
    return satisfied
  
        
    
if __name__ == "__main__":
    SPLIT = "evaluation"
    MIDI_DIR = f"../midi/midi_files/{SPLIT}/midi"
    AMOUNT = 50
    counter = 0
    satisfied_midi_file_paths = []
    midi_file_paths = [os.path.join(MIDI_DIR, f) 
                       for f in os.listdir(MIDI_DIR) if f.endswith('.mid')]#[:AMOUNT]

    os.makedirs("../vis_results", exist_ok=True)
    os.makedirs(f"../vis_results/{SPLIT}", exist_ok=True)
    
    for mfp in tqdm(midi_file_paths):
        if plot_midi_piano_roll(mfp, SPLIT):
            satisfied_midi_file_paths.append(mfp)

            counter += 1
            if counter == AMOUNT:
                break
            
    with open(f"../info/{SPLIT}_midi_file_paths_satisfied.txt", "w") as f:
        for mfp in satisfied_midi_file_paths:
            f.write(mfp + "\n")
            
    print(f"Saved {len(satisfied_midi_file_paths)} satisfied MIDI files")