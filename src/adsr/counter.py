import os
import wave
import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def process_wav_file(wavfile):
    with wave.open(wavfile, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return (duration, wavfile)


path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset"
json_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset/metadata.json"



# # Get list of all wav files
# wav_files = glob.glob(os.path.join(path, "*.wav"))

# # Create a pool of workers
# pool = mp.Pool(processes=mp.cpu_count())

# # Process files in parallel with progress bar
# results = list(tqdm(pool.imap(process_wav_file, wav_files), total=len(wav_files)))

# # Close the pool
# pool.close()
# pool.join()

# # Find maximum duration and corresponding file
# max_length, max_file = max(results, key=lambda x: x[0])

# print(f"Longest wav file: {max_file}")
# print(f"Duration: {max_length:.2f} seconds")
