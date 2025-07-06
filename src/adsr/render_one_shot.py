"""
This program has to be done in Buffett's macbook with serum plugins.
"""

import logging
import os
import json
import random
import dawdreamer as daw

from pathlib import Path
from glob import glob
from scipy.io import wavfile
from tqdm import tqdm


def calculate_note_duration(bars=1, bpm=120, time_signature=4):
    return bars * time_signature * 60 / bpm


def main(plugin_path, preset_dir, sample_rate=44100, bpm=120, 
         bars=1, padding=2, output_dir='output', logging_level='INFO', remove_pluck=True):

    # Create logger
    logging.basicConfig()
    logger = logging.getLogger('dawdreamer')
    logger.setLevel(logging_level.upper())

    # Get all preset paths
    preset_paths = list(glob(str(Path(preset_dir) / '**' / '*.fxp'), recursive=True)) #[:3]
    random.shuffle(preset_paths)
    logger.info(f"Found {len(preset_paths)} presets")
    
    
    # Remove pluck presets
    if remove_pluck:
        for pp in preset_paths:
            if 'pluck' in pp.lower() or 'PL' in pp:
                preset_paths.remove(pp)
        logger.info(f"After removing Pluck presets, we have {len(preset_paths)} presets")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate note duration (1 bars at 120 BPM = 2 seconds)
    note_duration = calculate_note_duration(bars, bpm)
    render_duration = note_duration + padding  # Add 2 seconds for release tail
    
    logger.info(f'Note duration: {note_duration} seconds')
    logger.info(f'Output directory: {output_dir}')

    # Initialize DawDreamer engine
    engine = daw.RenderEngine(sample_rate, block_size=512)
    engine.set_bpm(bpm)

    # Create synth processor
    synth = engine.make_plugin_processor("synth", plugin_path)
    graph = [(synth, [])]
    engine.load_graph(graph)

    # MIDI note numbers for C1 ~ C7
    # base_notes = {'2': 36, '3': 48, '4': 60}
    # click = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    # c_notes = {f'{note}{octave}': click[note] + base_notes[octave] for note in click.keys() for octave in base_notes.keys()}
    c_notes = {'C3': 48}

    preset_to_id = {}
    id_to_preset = {}
    for id, preset_path in enumerate(preset_paths):
        preset_name = Path(preset_path).stem
        preset_to_id[preset_name] = id
        id_to_preset[id] = preset_name
        
    os.makedirs('../info', exist_ok=True)
    with open('../info/preset_to_id.json', 'w') as f:
        json.dump(preset_to_id, f, indent=4, ensure_ascii=False)
    with open('../info/id_to_preset.json', 'w') as f:
        json.dump(id_to_preset, f, indent=4, ensure_ascii=False)

    # Initialize metadata collection
    metadata = {
        "dataset_info": {
            "total_files": 0,
            "sample_rate": sample_rate,
            "bpm": bpm,
            "note_duration": note_duration,
            "render_duration": render_duration,
            "velocity": 100,
            "notes_rendered": list(c_notes.keys()),
            "preset_directory": preset_dir,
            "output_directory": output_dir
        },
        "files": []
    }

    # Params
    velocity = 100
    time_seconds = 0.0

    # Process each preset
    for preset_path in tqdm(preset_paths, desc="Processing presets"):
        preset_name = Path(preset_path).stem
        preset_id = preset_to_id[preset_name]
        

        # Load the preset
        synth.load_preset(preset_path)
        
        # Generate audio for each C note
        note_id = 0
        for note_name, note_number in c_notes.items():
            # Clear any previous MIDI
            synth.clear_midi()
            
            # Add single MIDI note
            synth.add_midi_note(note_number, velocity, time_seconds, note_duration)
            
            # Render audio
            engine.render(render_duration)
            audio = engine.get_audio().mean(axis=0)
            
            # Save audio file
            output_filename = f"T{preset_id}_C{note_name}.wav"
            output_path = Path(output_dir) / output_filename
            wavfile.write(str(output_path), sample_rate, audio)
            
            # Add to metadata
            file_metadata = {
                "filename": output_filename,
                "timbre_id": preset_id,
                "content_id": note_id,
                "preset_name": preset_name,
            }
            
            metadata["files"].append(file_metadata)
            metadata["dataset_info"]["total_files"] += 1
            note_id += 1
                
            

    # Save comprehensive metadata
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f'All done! Metadata saved to {metadata_path}')



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate single note renderings for each Serum preset")
    parser.add_argument('--plugin', required=True, help="Path to Serum plugin (.dll, .vst3)")
    parser.add_argument('--preset-dir', required=True, help="Directory path of Serum presets")
    parser.add_argument('--sample-rate', default=44100, type=int, help="Sample rate")
    parser.add_argument('--bpm', default=120, type=float, help="Beats per minute")
    parser.add_argument('--bars', default=1, type=float, help="Number of bars for each note")
    parser.add_argument('--padding', default=2, type=float, help="Padding in seconds")
    parser.add_argument('--output-dir', default='output', help="Output directory")
    parser.add_argument('--remove-pluck', type=bool, default=False, help="Remove Pluck presets")
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], 
                       help="Logger level")
    
    args = parser.parse_args()

    main(args.plugin, args.preset_dir, args.sample_rate, args.bpm, 
         args.bars, args.padding, args.output_dir, args.log_level, args.remove_pluck)
