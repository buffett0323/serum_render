import dawdreamer as daw
import json
import random
from pathlib import Path
from scipy.io import wavfile


def explore_parameters(synth):
    """Explore and print all available parameters of the loaded plugin"""
    print("\n=== Available Parameters ===")
    param_count = synth.get_plugin_parameter_size()
    print(f"Total parameters: {param_count}")
    
    # Look for ADSR-related parameters
    adsr_params = []
    store_all_params = {}
    
    for i in range(param_count):
        param_name = synth.get_parameter_name(i)
        param_value = synth.get_parameter(i)
        
        # Check if parameter name contains ADSR keywords
        if any(keyword in param_name.lower() for keyword in ['attack', 'decay', 'sustain', 'release', 'adsr', 'env']):
            adsr_params.append({
                'index': i,
                'name': param_name, 
                'value': param_value
            })
            print(f"  ADSR-related [{i}]: {param_name} = {param_value}")

        store_all_params[param_name] = {
            'value': param_value,
        }
        
    
    with open('adsr_modified_samples/all_params.json', 'w') as f:
        json.dump(store_all_params, f, indent=4, ensure_ascii=False)
    
    return adsr_params


def modify_adsr_example(synth, preset_name):
    """Example function showing how to modify ADSR parameters"""
    print(f"\n=== Modifying ADSR for preset: {preset_name} ===")
    
    # First, let's explore what parameters are available
    adsr_params = explore_parameters(synth)
    
    if not adsr_params:
        print("No ADSR parameters found. This might be because:")
        print("1. Parameter names don't contain obvious ADSR keywords")
        print("2. ADSR is controlled by different parameter names")
        print("3. Need to explore all parameters manually")
        return False
    
    # Example: Try to modify common ADSR parameter names
    # Note: Serum might use different parameter names, so you may need to adjust these
    adsr_modifications = {
        'env1 atk': random.uniform(0.0, 0.5),   # Random attack 0-0.5
        'env1 dec': random.uniform(0.1, 0.8),    # Random decay 0.1-0.8  
        'env1 sus': random.uniform(0.3, 0.9),  # Random sustain 0.3-0.9
        'env1 rel': random.uniform(0.2, 1.0)   # Random release 0.2-1.0
    }
    
    
    modified_params = {}
    
    # Try to find and modify ADSR parameters
    for target_name, new_value in adsr_modifications.items():
        found = False
        for param in adsr_params:
            param_name_lower = param['name'].lower()
            
            # Check if this parameter matches our target ADSR component
            if target_name in param_name_lower:
                old_value = synth.get_parameter(param['index'])
                synth.set_parameter(param['index'], new_value)
                actual_value = synth.get_parameter(param['index'])
                
                modified_params[param['name']] = {
                    'old': old_value,
                    'target': new_value, 
                    'actual': actual_value
                }
                
                print(f"  Modified {param['name']}: {old_value:.3f} -> {actual_value:.3f}")
                found = True
                break
        
        if not found:
            print(f"  Could not find parameter for: {target_name}")
    
    return modified_params


def main():
    # Configuration
    SAMPLE_RATE = 44100
    PLUGIN_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"  # Adjust path as needed
    PRESET_PATH = "adsr_modified_samples/test.fxp" #"/Users/buffettliu/Desktop/Music_AI/Codes/render_dataset/fxp_preset/train/lead/LD - Arp Attack.fxp"    # Adjust path as needed
    OUTPUT_DIR = Path("adsr_modified_samples")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize engine
    engine = daw.RenderEngine(SAMPLE_RATE, block_size=512)
    engine.set_bpm(120)
    
    # Create synth processor
    synth = engine.make_plugin_processor("synth", PLUGIN_PATH)
    engine.load_graph([(synth, [])])
    
    print("=== Serum ADSR Parameter Modifier ===")
    
    # Load a preset
    print(f"Loading preset: {PRESET_PATH}")
    synth.load_preset(PRESET_PATH)
    preset_name = Path(PRESET_PATH).stem
    
    # Render original sound for comparison
    print("\n=== Rendering Original ===")
    synth.clear_midi()
    synth.add_midi_note(60, 100, 0.0, 1.0)  # C4 for 1 second
    engine.render(2.0)  # 1 second total (1 note + 0 release)
    original_audio = engine.get_audio()
    
    # Save original
    original_path = OUTPUT_DIR / f"{preset_name}_original.wav"
    wavfile.write(str(original_path), SAMPLE_RATE, original_audio.transpose())
    print(f"Saved original: {original_path}")
    
    
    # Modify ADSR parameters
    modified_params = modify_adsr_example(synth, preset_name)
    
    # Render modified sound
    print("\n=== Rendering Modified ===")
    synth.clear_midi()
    synth.add_midi_note(60, 100, 0.0, 1.0)  # Same note
    engine.render(2.0)
    modified_audio = engine.get_audio()
    
    # Save modified
    modified_path = OUTPUT_DIR / f"{preset_name}_modified.wav"
    wavfile.write(str(modified_path), SAMPLE_RATE, modified_audio.transpose())
    print(f"Saved modified: {modified_path}")
    
    # Save parameter changes to JSON
    if modified_params:
        json_path = OUTPUT_DIR / f"{preset_name}_parameter_changes.json"
        with open(json_path, 'w') as f:
            json.dump(modified_params, f, indent=4, ensure_ascii=False)
        print(f"Saved parameter changes: {json_path}")
    
    print("\n=== Done! ===")
    print("Compare the original and modified audio files to hear the difference.")
    print("Check the JSON file to see which parameters were changed.")


if __name__ == "__main__":
    main() 