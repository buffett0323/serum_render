import tkinter as tk
from tkinter import simpledialog
from pedalboard import load_plugin

SERUM_PLUGIN_PATH = "/Library/Audio/Plug-Ins/Components/Serum.component" # "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
PLUGIN_NAME = "Serum" #"Serum 2" # "Serum 2 FX"


if __name__ == "__main__":
    
    # Load Serum plugin
    serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)
    
    # Show the GUI
    serum.show_editor()

    # Get preset name
    preset_name = input("Enter a name for the preset:")
    
    # Mapping
    preset_path = ""
    if "LD" in preset_name:
        preset_path = "lead"
    elif "BA" in preset_name:
        preset_path = "bass"
    elif "PAD" in preset_name:
        preset_path = "pad"
    elif "PL" in preset_name:
        preset_path = "pluck"
    else:
        preset_path = "keys"
    
    # Check if the user provided a name
    if preset_name:
        # Save the current plugin state to a .vstpreset file
        with open(f'../vstpreset/{preset_path}/{preset_name}.vstpreset', 'wb') as f:
            f.write(serum.raw_state)
        print(f"Preset saved as {preset_name}.vstpreset")
    else:
        print("No preset name provided. Preset not saved.")