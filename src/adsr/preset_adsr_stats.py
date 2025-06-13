import dawdreamer as daw
import numpy as np
import json
import random
import os
from glob import glob
from pathlib import Path
from scipy.io import wavfile
from tqdm import tqdm

def explore_parameters(synth):
    param_count = synth.get_plugin_parameter_size()
    
    # Look for ADSR-related parameters
    adsr_params = []
    
    for i in range(param_count):
        param_name = synth.get_parameter_name(i)
        param_value = synth.get_parameter(i)
        
        # Check if parameter name contains ADSR keywords
        if any(keyword in param_name.lower() for keyword in ['attack', 'decay', 'sustain', 'release', 'adsr', 'env']):
            if "env1" in param_name.lower():
                adsr_params.append({
                    'name': param_name, 
                    'value': param_value
                })
                
    return adsr_params



def main():
    # Configuration
    SAMPLE_RATE = 44100
    PLUGIN_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"  # Adjust path as needed
    PRESET_PATH = "../../fxp_preset/train" #"/Users/buffettliu/Desktop/Music_AI/Codes/render_dataset/fxp_preset/train/lead/LD - Arp Attack.fxp"    # Adjust path as needed
    STEMS = ["lead", "keys", "pad", "pluck", "synth", "vox"]
    
    # Get folder mapping
    folder_mapping = {}
    all_folders = os.listdir(PRESET_PATH)
    for folder in all_folders:
        for stem in STEMS:
            if stem in folder:
                folder_mapping[folder] = stem
                break
           
    
    assert len(folder_mapping) == len(all_folders), "Folder mapping is not complete"
    
    env_items = ["Env1 Atk", "Env1 Dec", "Env1 Hold", "Env1 Sus", "Env1 Rel"]
    env_stats = {stem: {item: [] for item in env_items}
                 for stem in STEMS}
    preset_paths = list(glob(str(Path(PRESET_PATH) / '**' / '*.fxp'), recursive=True))
    random.shuffle(preset_paths)

    # Processing
    for preset_path in tqdm(preset_paths):
        
        # Get stem
        folder = preset_path.split("/")[-2]
        stem = folder_mapping[folder]

        # Initialize engine
        engine = daw.RenderEngine(SAMPLE_RATE, block_size=512)
        engine.set_bpm(120)
        
        # Create synth processor
        synth = engine.make_plugin_processor("synth", PLUGIN_PATH)
        engine.load_graph([(synth, [])])

        synth.load_preset(preset_path)

        preset_params = explore_parameters(synth)

        for dic in preset_params:
            if dic['name'] in env_items:
                env_stats[stem][dic['name']].append(dic['value'])
                

    with open("stats/env_stats.json", "w") as f:
        json.dump(env_stats, f, indent=4)


    # Process and Save stats
    final_env_stats = {}
    for stem in env_stats:
        for item in env_stats[stem]:
            if len(env_stats[stem][item]) > 0:
                final_env_stats[f"{stem}, {item}"] = {
                    "mean": np.mean(env_stats[stem][item]),
                    "std": np.std(env_stats[stem][item])
                }


    with open("stats/final_env_stats.json", "w") as f:
        json.dump(final_env_stats, f, indent=4)



if __name__ == "__main__":
    main() 