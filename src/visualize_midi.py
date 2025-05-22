import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_midi_piano_roll(midi_path, max_duration=60):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    fs = 100  # Time resolution in frames per second
    piano_roll = midi_data.get_piano_roll(fs=fs)

    end_frame = min(piano_roll.shape[1], int(max_duration * fs))

    plt.figure(figsize=(14, 6))
    plt.imshow(piano_roll[:, :end_frame], aspect='auto', origin='lower', cmap='gray_r')
    plt.xlabel("Time (frames)")
    plt.ylabel("MIDI note number")
    plt.title(f"Piano Roll of: {midi_path}")
    plt.colorbar(label='Velocity')
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    MIDI_DIR = "../midi/midi_files/evaluation"
    MIDI_FILE_PATH = os.path.join(MIDI_DIR, os.listdir(MIDI_DIR)[0])
    
    plot_midi_piano_roll(MIDI_FILE_PATH)