import dawdreamer as daw
import json
from pathlib import Path


def deep_parameter_scan(synth, preset_name):
    """Comprehensive scan of ALL parameters, looking for hidden LFO mode controls"""
    print(f"\n=== Deep Parameter Scan for: {preset_name} ===")
    
    param_count = synth.get_plugin_parameter_size()
    print(f"Total parameters found: {param_count}")
    
    all_params = {}
    lfo_related = {}
    potential_modes = {}
    
    # Scan every single parameter
    for i in range(param_count):
        try:
            param_name = synth.get_parameter_name(i)
            param_value = synth.get_parameter(i)
            param_text = synth.get_parameter_text(i)  # This might show discrete values!
            
            param_info = {
                'index': i,
                'value': param_value,
                'text': param_text,
                'name': param_name
            }
            
            all_params[param_name] = param_info
            
            # Look for LFO-related parameters
            if any(keyword in param_name.lower() for keyword in ['lfo', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8']):
                lfo_related[param_name] = param_info
            
            # Look for potential mode parameters
            if any(keyword in param_name.lower() for keyword in ['mode', 'trig', 'env', 'state', 'type', 'switch']):
                potential_modes[param_name] = param_info
                
            # Look for parameters with discrete text values
            if param_text and param_text != str(param_value):
                if param_name not in potential_modes:
                    potential_modes[param_name] = param_info
                    
        except Exception as e:
            print(f"Error reading parameter {i}: {e}")
    
    print(f"\nLFO-related parameters found: {len(lfo_related)}")
    for name, info in lfo_related.items():
        print(f"  [{info['index']}] {name}: {info['value']} (text: '{info['text']}')")
    
    print(f"\nPotential mode/discrete parameters: {len(potential_modes)}")
    for name, info in potential_modes.items():
        print(f"  [{info['index']}] {name}: {info['value']} (text: '{info['text']}')")
    
    # Save comprehensive data
    output_data = {
        'preset_name': preset_name,
        'total_params': param_count,
        'all_parameters': all_params,
        'lfo_related': lfo_related,
        'potential_modes': potential_modes
    }
    
    output_file = f"deep_scan_{preset_name}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDeep scan saved to: {output_file}")
    return output_data


def compare_presets_for_lfo_differences(plugin_path, preset1_path, preset2_path):
    """Compare two presets to find parameters that change when LFO modes differ"""
    print("=== Comparing Presets for LFO Mode Differences ===")
    
    engine = daw.RenderEngine(44100, block_size=512)
    synth = engine.make_plugin_processor("synth", plugin_path)
    engine.load_graph([(synth, [])])
    
    # Scan first preset
    synth.load_preset(preset1_path)
    scan1 = deep_parameter_scan(synth, Path(preset1_path).stem)
    
    # Scan second preset
    synth.load_preset(preset2_path)
    scan2 = deep_parameter_scan(synth, Path(preset2_path).stem)
    
    # Find differences
    differences = {}
    for param_name in scan1['all_parameters']:
        if param_name in scan2['all_parameters']:
            val1 = scan1['all_parameters'][param_name]['value']
            val2 = scan2['all_parameters'][param_name]['value']
            text1 = scan1['all_parameters'][param_name]['text']
            text2 = scan2['all_parameters'][param_name]['text']
            
            if abs(val1 - val2) > 0.001 or text1 != text2:
                differences[param_name] = {
                    'preset1': {'value': val1, 'text': text1},
                    'preset2': {'value': val2, 'text': text2},
                    'index': scan1['all_parameters'][param_name]['index']
                }
    
    print(f"\nParameters that differ between presets: {len(differences)}")
    for name, diff in differences.items():
        print(f"  {name}: {diff['preset1']['value']} ('{diff['preset1']['text']}') → {diff['preset2']['value']} ('{diff['preset2']['text']}')")
    
    return differences


def investigate_parameter_ranges(synth):
    """Investigate parameter ranges to find discrete/enum parameters"""
    print("\n=== Investigating Parameter Ranges ===")
    
    param_count = synth.get_plugin_parameter_size()
    discrete_params = {}
    
    for i in range(param_count):
        try:
            param_name = synth.get_parameter_name(i)
            
            # Try to get parameter range information (if available)
            try:
                # Some dawdreamer versions might have get_parameter_range
                if hasattr(synth, 'get_parameter_range'):
                    range_info = synth.get_parameter_range(i)
                    print(f"  {param_name}: range = {range_info}")
            except:
                pass
            
            # Check if parameter has limited discrete values by testing different values
            original_value = synth.get_parameter(i)
            test_values = [0.0, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0]
            actual_values = []
            
            for test_val in test_values:
                synth.set_parameter(i, test_val)
                actual_val = synth.get_parameter(i)
                actual_text = synth.get_parameter_text(i)
                if actual_val not in actual_values:
                    actual_values.append((actual_val, actual_text))
            
            # Restore original value
            synth.set_parameter(i, original_value)
            
            # If we got fewer unique values than we tested, it might be discrete
            if len(actual_values) < len(test_values):
                discrete_params[param_name] = {
                    'index': i,
                    'possible_values': actual_values,
                    'original_value': original_value
                }
                print(f"  DISCRETE: {param_name} has {len(actual_values)} possible values: {actual_values}")
                
        except Exception as e:
            print(f"Error investigating parameter {i}: {e}")
    
    return discrete_params


if __name__ == "__main__":
    PLUGIN_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"
    PRESET_PATH = "adsr_modified_samples/test.fxp"
    
    # Initialize
    engine = daw.RenderEngine(44100, block_size=512)
    synth = engine.make_plugin_processor("synth", PLUGIN_PATH)
    engine.load_graph([(synth, [])])
    synth.load_preset(PRESET_PATH)
    
    # Deep parameter scan
    deep_scan_results = deep_parameter_scan(synth, Path(PRESET_PATH).stem)
    
    # Investigate parameter ranges for discrete values
    discrete_params = investigate_parameter_ranges(synth)
    
    print(f"\nFound {len(discrete_params)} potentially discrete parameters")
    
    # Save discrete parameter findings
    with open("discrete_parameters.json", 'w') as f:
        json.dump(discrete_params, f, indent=2, ensure_ascii=False) 