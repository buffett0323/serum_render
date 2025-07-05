# 
# This file is part of the DawDreamer distribution (https://github.com/DBraun/DawDreamer).
# Copyright (c) 2023 David Braun.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import logging
import multiprocessing
import time
import traceback
from collections import namedtuple
from glob import glob
import os
from pathlib import Path
import json

# extra libraries to install with pip
import dawdreamer as daw
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# Buffett Added
import pretty_midi
from itertools import product

Item = namedtuple("Item", "preset_path midi_path timbre_id content_id")


class Worker:

    def __init__(
        self, 
        queue: multiprocessing.Queue, 
        plugin_path: str,
        sample_rate=44100, 
        block_size=512, 
        bpm=120, 
        render_duration=10, 
        output_dir='output'):
        self.queue = queue
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.bpm = bpm
        self.plugin_path = plugin_path
        self.render_duration = render_duration
        self.output_dir = Path(output_dir)

    def startup(self):
        engine = daw.RenderEngine(self.sample_rate, self.block_size)
        engine.set_bpm(self.bpm)

        synth = engine.make_plugin_processor("synth", self.plugin_path)
        
        graph = [(synth, [])]
        engine.load_graph(graph)

        self.engine = engine
        self.synth = synth

    def peak_normalize(self, audio):
        """Peak normalize audio to -1..1 range to prevent clipping"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def process_item(self, item: Item):
        preset_path = item.preset_path
        midi_path = item.midi_path
        timbre_id = item.timbre_id
        content_id = item.content_id
        
        self.synth.load_preset(preset_path)

        midi_data = pretty_midi.PrettyMIDI(midi_path)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start_time = note.start
                duration = note.end - note.start
                pitch = note.pitch
                velocity = note.velocity if note.velocity > 80 else 80
                self.synth.add_midi_note(pitch, velocity, start_time, duration)

        self.engine.render(self.render_duration)
        audio = self.engine.get_audio().mean(axis=0)
        
        # Apply peak normalization
        audio = self.peak_normalize(audio)
        
        # Create filename with T000_C000 format
        output_filename = f'T{timbre_id:03d}_C{content_id:03d}.wav'
        output_path = self.output_dir / output_filename
        wavfile.write(str(output_path), self.sample_rate, audio)

        self.synth.clear_midi()


    def run(self):
        try:
            self.startup()
            while True:
                try:
                    item = self.queue.get_nowait()
                    self.process_item(item)
                except multiprocessing.queues.Empty:
                    break
        except Exception as e:
            return traceback.format_exc()


def main(plugin_path, preset_dir, sample_rate=44100, bpm=120, 
    render_duration=4, num_workers=None,
    output_dir='output', logging_level='INFO', split='train'):

    # Create logger
    logging.basicConfig()
    logger = logging.getLogger('dawdreamer')
    logger.setLevel(logging_level.upper())

    # Get all preset paths
    preset_paths = list(glob(str(Path(preset_dir)/ '**' / '*.fxp'), recursive=True))

    # Load all MIDI file paths
    with open(f"../info/{split}_midi_file_paths_satisfied.txt", "r") as f:
        midi_file_paths = [line.strip() for line in f.readlines()][:10] #[:100]

    # Create ID mappings
    # Timbre mapping
    with open('info/preset_to_id.json', 'r') as f:
        preset_to_id = json.load(f)

    timbre_id_map = {}
    for preset_path in preset_paths:
        preset_name = Path(preset_path).stem
        timbre_id_map[preset_name] = preset_to_id[preset_name]

    # Content mapping
    content_id_map = {Path(midi_path).stem: i for i, midi_path in enumerate(midi_file_paths)}
    
    # Save metadata
    metadata = {
        'timbre_id_map': {str(k): v for k, v in timbre_id_map.items()},
        'content_id_map': {str(k): v for k, v in content_id_map.items()},
        'total_timbres': len(timbre_id_map),
        'total_contents': len(content_id_map),
        'sample_rate': sample_rate,
        'bpm': bpm,
        'render_duration': render_duration,
        'split': split
    }
    
    metadata_path = Path(output_dir) / 'metadata.json'
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info(f"Total timbres: {len(timbre_id_map)}, Total contents: {len(content_id_map)}")


    # Create all combinations of presets and MIDI files
    all_combinations = list(product(preset_paths, midi_file_paths))
    num_items = len(all_combinations)
    logger.info(f"Total combinations (preset x midi): {num_items}")

    # Create input queue and fill it with ID information
    input_queue = multiprocessing.Manager().Queue()
    for preset_path, midi_path in all_combinations:
        timbre_id = timbre_id_map[Path(preset_path).stem]
        content_id = content_id_map[Path(midi_path).stem]
        input_queue.put(Item(
            preset_path=preset_path, 
            midi_path=midi_path,
            timbre_id=timbre_id,
            content_id=content_id
        ))

    # Determine number of workers
    num_processes = num_workers or multiprocessing.cpu_count()
    logger.info(f'Render duration: {render_duration}')
    logger.info(f'Using num workers: {num_processes}')
    logger.info(f'Output directory: {output_dir}')

    # Start multiprocessing
    workers = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i in range(num_processes):
            worker = Worker(
                input_queue,
                plugin_path,
                sample_rate=sample_rate,
                bpm=bpm,
                render_duration=render_duration,
                output_dir=output_dir
            )
            async_result = pool.apply_async(worker.run)
            workers.append(async_result)

        # Progress bar
        pbar = tqdm(total=num_items)
        while True:
            incomplete_count = sum(1 for w in workers if not w.ready())
            pbar.update(pbar.total - incomplete_count - pbar.n)
            if incomplete_count == 0:
                break
            time.sleep(0.1)
        pbar.close()

    # Log any worker errors
    for i, worker in enumerate(workers):
        exception = worker.get()
        if exception is not None:
            logger.error(f"Exception in worker {i}:\n{exception}")

    logger.info('All done!')

if __name__ == "__main__":
    # We're using multiprocessing.Pool, so our code MUST be inside __main__.
    # See https://docs.python.org/3/library/multiprocessing.html

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', required=True, help="Path to plugin instrument (.dll, .vst3).")
    parser.add_argument('--preset-dir', required=True, help="Directory path of plugin presets.")
    parser.add_argument('--sample-rate', default=44100, type=int, help="Sample rate for the plugin.")
    parser.add_argument('--bpm', default=120, type=float, help="Beats per minute for the Render Engine.")
    parser.add_argument('--render-duration', default=4, type=float, help="Render duration in seconds.")
    parser.add_argument('--num-workers', default=None, type=int, help="Number of workers to use.")
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__),'output'), help="Output directory.")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL', 'NOTSET'], help="Logger level.")
    parser.add_argument('--split', default='train', choices=['train', 'evaluation'], help="Split to render.")
    args = parser.parse_args()
    print(args)
    
    main(args.plugin, args.preset_dir, args.sample_rate, args.bpm, 
        args.render_duration, args.num_workers, args.output_dir, args.log_level, args.split)
