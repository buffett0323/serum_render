from pedalboard import load_plugin
import os

SERUM_PLUGIN_PATH = "/Library/Audio/Plug-Ins/Components/Serum.component" # "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
PLUGIN_NAME = "Serum" #"Serum 2" # "Serum 2 FX"
SERUM_PRESET_DIR = "../serum_preset"
STEM = "pluck"

from process import stem_mapping
STEM_PREFIX = stem_mapping[STEM]

def select_preset_cli(preset_dir='preset'):
    # Ensure the preset directory exists
    if not os.path.isdir(preset_dir):
        print(f"Error: The directory '{preset_dir}' does not exist.")
        return None

    # List all preset files
    files = [f for f in os.listdir(preset_dir) if os.path.isfile(os.path.join(preset_dir, f))]
    if not files:
        print(f"No preset files found in '{preset_dir}'.")
        return None
    
    # Filter prefix
    selected_files = []
    while len(selected_files) == 0:
        prefix = input("Enter the prefix of the preset you want to select: ")
        selected_files = [f for f in files if prefix in f]

        # Display the list of presets
        print("Available Presets:")
        for idx, file in enumerate(selected_files, start=1):
            print(f"{idx}. {file}")

    # Prompt user for selection
    while True:
        try:
            choice = int(input("Enter the number of the preset you want to select: "))
            if 1 <= choice <= len(selected_files):
                selected_file = selected_files[choice - 1]
                print(f"You have selected: {selected_file}")
                return selected_file
            else:
                print("Invalid selection. Please enter a number corresponding to the presets listed.")
        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    
    # Load Serum plugin
    serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)
    
    # Show the GUI
    serum.show_editor()


    # Get the preset name from the user
    preset_name = select_preset_cli(os.path.join(SERUM_PRESET_DIR, STEM))
    preset_name = preset_name.split(".fxp")[0].split("- ")[-1]
    
    # Check if the user provided a name
    if preset_name:
        # Save the current plugin state to a .vstpreset file
        with open(f'../vstpreset/{STEM}/{STEM_PREFIX} - {preset_name}.vstpreset', 'wb') as f:
            f.write(serum.raw_state)
        print(f"Preset saved as {preset_name}.vstpreset")
    else:
        print("No preset name provided. Preset not saved.")
