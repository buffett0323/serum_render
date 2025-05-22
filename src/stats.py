import pretty_midi
import os
from tqdm import tqdm


def get_midi_duration(midi_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data.get_end_time()
    except Exception as e:
        print(f"Failed to read {midi_path}: {e}")
        return 0.0  # Return 0 if the file is unreadable

if __name__ == "__main__":
    MIDI_DIR = "../midi/midi_files/evaluation/midi/"
    durations = []

    for file in tqdm(os.listdir(MIDI_DIR)):
        if file.endswith('.mid') or file.endswith('.midi'):
            path = os.path.join(MIDI_DIR, file)
            duration = get_midi_duration(path)
            durations.append(duration)
            print(f"{file}: {duration:.2f} seconds")

    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\nAverage MIDI duration: {avg_duration:.2f} seconds")
    else:
        print("No valid MIDI files found.")
        
    # 88.66 seconds