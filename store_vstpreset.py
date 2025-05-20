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
    
    
    # Create a hidden root window for the dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user for the preset name
    preset_name = simpledialog.askstring("Preset Name", "Enter a name for the preset:")

    
    # Check if the user provided a name
    if preset_name:
        # Save the current plugin state to a .vstpreset file
        with open(f'vstpreset/{preset_name}.vstpreset', 'wb') as f:
            f.write(serum.raw_state)
        print(f"Preset saved as {preset_name}.vstpreset")
    else:
        print("No preset name provided. Preset not saved.")