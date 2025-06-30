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
    total_beats = bars * time_signature
    beats_per_second = bpm / 60
    return total_beats / beats_per_second


def main(plugin_path, preset_dir, sample_rate=44100, bpm=120, 
         bars=4, output_dir='output', logging_level='INFO'):

    # Create logger
    logging.basicConfig()
    logger = logging.getLogger('dawdreamer')
    logger.setLevel(logging_level.upper())

    # Get all preset paths
    preset_paths = list(glob(str(Path(preset_dir) / '**' / '*.fxp'), recursive=True)) #[:3]
    random.shuffle(preset_paths)
    logger.info(f"Found {len(preset_paths)} presets")
    
    
    # Remove pluck presets
    for pp in preset_paths:
        if 'pluck' in pp.lower() or 'PL' in pp:
            preset_paths.remove(pp)
    logger.info(f"After removing, we have {len(preset_paths)} presets")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate note duration (1 bars at 120 BPM = 2 seconds)
    note_duration = calculate_note_duration(bars, bpm)
    render_duration = note_duration + 2.0  # Add 2 seconds for release tail
    
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
    c_notes = {
        # 'C1': 24,
        # 'C2': 36, 
        # 'C3': 48,
        'C4': 60,
        # 'C5': 72,
        # 'C6': 84,
    }

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

    # Params
    velocity = 100
    time_seconds = 0.0


    # Process each preset
    for preset_path in tqdm(preset_paths, desc="Processing presets"):
        preset_name = Path(preset_path).stem
        
        try:
            # Load the preset
            synth.load_preset(preset_path)
            
            # Generate audio for each C note
            for note_name, note_number in c_notes.items():
                # Clear any previous MIDI
                synth.clear_midi()
                
                # Add single MIDI note
                synth.add_midi_note(note_number, velocity, time_seconds, note_duration)
                
                # Render audio
                engine.render(render_duration)
                audio = engine.get_audio()
                
                # Save audio file
                output_filename = f"T{preset_to_id[preset_name]}_{note_name}.wav"
                output_path = Path(output_dir) / output_filename
                wavfile.write(str(output_path), sample_rate, audio.transpose())
                
        
        except Exception as e:
            logger.error(f"Error processing preset {preset_name}: {e}")
            continue


    logger.info('All done!')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate single note renderings for each Serum preset")
    parser.add_argument('--plugin', required=True, help="Path to Serum plugin (.dll, .vst3)")
    parser.add_argument('--preset-dir', required=True, help="Directory path of Serum presets")
    parser.add_argument('--sample-rate', default=44100, type=int, help="Sample rate")
    parser.add_argument('--bpm', default=120, type=float, help="Beats per minute")
    parser.add_argument('--bars', default=1, type=int, help="Number of bars for each note")
    parser.add_argument('--output-dir', default='output', help="Output directory")
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], 
                       help="Logger level")
    
    args = parser.parse_args()

    main(args.plugin, args.preset_dir, args.sample_rate, args.bpm, 
         args.bars, args.output_dir, args.log_level)
