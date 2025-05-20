from pedalboard import load_plugin, AudioUnitPlugin 
import pedalboard
import soundfile as sf
import numpy as np
import mido # For MIDI parsing
import json # For preset parameters
import time # For progress indication

# --- Configuration ---
SERUM_PLUGIN_PATH = "/Library/Audio/Plug-Ins/Components/Serum.component" # "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
PLUGIN_NAME = "Serum" #"Serum 2" # "Serum 2 FX"
# FXP_PRESET_PATH = "serum preset/lead/LD - Starboy Era.fxp" 
MIDI_INPUT_FILE_PATH = "piano.mid"  # <-- CHANGE THIS
AUDIO_OUTPUT_FILE_PATH = "output_serum_audio.wav" # <-- CHANGE THIS
PRESET_DATA_FILE_PATH = "jsons/serum_parameters2.json"

# Audio Rendering Settings
SAMPLE_RATE = 44100.0  # Hz
BUFFER_SIZE = 512      # Samples per processing block (power of 2 often good)
OUTPUT_CHANNELS = 2    # 1 for mono, 2 for stereo (Serum often outputs stereo)
TAIL_DURATION_SECONDS = 3.0 # Extra time to capture release tails after MIDI ends



# --- Helper Functions (from previous script, slightly adapted) ---    
def load_serum_plugin(plugin_path):
    """Loads the Serum plugin using pedalboard."""
    try:
        serum = load_plugin(plugin_path, plugin_name=PLUGIN_NAME)
        print(f"Successfully loaded {PLUGIN_NAME}: {serum.name}")
        return serum
    except Exception as e:
        print(f"Error loading Serum plugin from '{plugin_path}': {e}")
        return None


def load_parameters_from_file(file_path):
    """Loads parameter data from a JSON file (conceptual)."""
    if not file_path:
        return None
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
            print(f"Successfully loaded parameter data from '{file_path}'.")
            return params
    except FileNotFoundError:
        print(f"Info: Preset data file not found at '{file_path}'. Using Serum's current state.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Ensure it's valid JSON.")
        return None
    except Exception as e:
        print(f"An error occurred loading parameters: {e}")
        return None


def apply_parameters_to_serum(serum_instance, parameters_to_set):
    """Attempts to set parameters on the loaded Serum instance (conceptual)."""
    if not serum_instance or not parameters_to_set:
        if parameters_to_set: # Only print if params were meant to be applied
             print("Serum instance not available or no parameters to set. Skipping parameter application.")
        return

    print("\nAttempting to apply parameters to Serum:")
    applied_count = 0
    failed_count = 0
    for param_name_from_file, value_from_file in parameters_to_set.items():
        param_name_in_serum_vst = param_name_from_file # Placeholder for mapping
        try:
            if param_name_in_serum_vst in serum_instance.parameters:
                param_object = serum_instance.parameters[param_name_in_serum_vst]
                target_value = float(value_from_file)
                clamped_value = max(param_object.min_value, min(target_value, param_object.max_value))
                param_object.value = clamped_value
                # print(f"  - Set '{param_name_in_serum_vst}' to {clamped_value:.4f}")
                applied_count += 1
            else:
                # print(f"  - Parameter '{param_name_in_serum_vst}' not found. Skipping.")
                failed_count += 1
        except Exception as e:
            print(f"  - Error setting parameter '{param_name_in_serum_vst}': {e}")
            failed_count += 1
    summary = f"Parameter application: {applied_count} applied"
    if failed_count > 0:
        summary += f", {failed_count} failed/skipped (อาจเป็นเพราะชื่อพารามิเตอร์ไม่ตรงกัน หรือค่าไม่ถูกต้อง)."
    print(summary)


# --- New MIDI and Rendering Functions ---
def get_midi_events_and_total_samples(midi_file_path, sample_rate):
    """
    Parses a MIDI file, converts messages to bytes with sample offsets.
    Returns a list of (sample_offset, midi_bytes) tuples, and total song duration in samples.
    """
    try:
        mid = mido.MidiFile(midi_file_path)
    except FileNotFoundError:
        print(f"Error: MIDI file not found at '{midi_file_path}'")
        return None, 0
    except Exception as e:
        print(f"Error opening MIDI file '{midi_file_path}': {e}")
        return None, 0

    print(f"Parsing MIDI file: {midi_file_path} (length: {mid.length:.2f} seconds)")
    
    events = []
    current_time_seconds = 0.0
    # Use a high default for ticks_per_beat if not found, though mido usually provides it.
    ticks_per_beat = mid.ticks_per_beat if mid.ticks_per_beat else 480
    # Default tempo (120 BPM) = 500,000 microseconds per beat
    current_tempo_us_per_beat = 500000

    # Process messages from all tracks, sorted by time
    # Note: mido.play() gives absolute time in seconds *considering tempo changes*.
    # However, pedalboard's `midi` parameter for `process()` takes raw bytes.
    # We will provide MIDI messages per block.
    # A simpler way is to collect all messages with their absolute sample offsets.

    absolute_tick = 0
    for msg in mido.merge_tracks(mid.tracks): # merge_tracks handles delta times
        # Convert delta time (ticks) to seconds based on current tempo
        delta_seconds = mido.tick2second(msg.time, ticks_per_beat, current_tempo_us_per_beat)
        current_time_seconds += delta_seconds
        absolute_tick += msg.time # msg.time is delta from previous event in merged track

        if msg.is_meta:
            if msg.type == 'set_tempo':
                current_tempo_us_per_beat = msg.tempo
            # We could filter other meta messages or pass them; Serum might ignore them.
            # For now, let's pass most non-meta messages.
            continue # Skip adding meta messages to the byte stream for the plugin for now

        if msg.type in ['note_on', 'note_off', 'control_change', 'pitchwheel', 'polytouch', 'aftertouch', 'program_change']:
            sample_offset = int(current_time_seconds * sample_rate)
            events.append((sample_offset, msg.bin()))

    # Sort events by sample_offset, just in case (though merge_tracks should be chronological)
    events.sort(key=lambda x: x[0])

    # Calculate total song duration in samples based on MIDI length
    total_midi_duration_samples = int(mid.length * sample_rate)
    
    # Add tail duration
    total_render_samples = total_midi_duration_samples + int(TAIL_DURATION_SECONDS * sample_rate)
    
    print(f"MIDI parsed. Found {len(events)} relevant events.")
    print(f"Total MIDI duration: {mid.length:.2f}s ({total_midi_duration_samples} samples).")
    print(f"Total render duration (with tail): {total_render_samples / sample_rate:.2f}s ({total_render_samples} samples).")
    
    return events, total_render_samples


def render_audio_from_midi(serum_instance, midi_events, total_samples,
                           sample_rate, buffer_size, num_channels):
    """
    Renders audio by processing MIDI events through Serum in blocks.
    """
    if not serum_instance:
        print("Serum instance not available for rendering.")
        return None

    print(f"\nStarting audio rendering: {total_samples / sample_rate:.2f} seconds...")
    
    # pedalboard uses (num_channels, num_frames)
    output_audio = np.zeros((num_channels, total_samples), dtype=np.float32)
    
    current_block_start_sample = 0
    midi_event_idx = 0
    num_blocks = (total_samples + buffer_size -1) // buffer_size # Ceiling division

    # Reset plugin state before processing (optional, but good practice for consistency)
    # try:
    #     serum_instance.reset()
    #     print("Serum plugin state reset.")
    # except Exception as e:
    #     print(f"Note: Could not call reset() on Serum plugin (might not be implemented by all wrappers/plugins): {e}")


    for i in range(num_blocks):
        block_start_time = time.time()
        
        # Input for an instrument plugin is typically silence
        # Shape: (num_channels, buffer_size)
        input_block_silence = np.zeros((num_channels, buffer_size), dtype=np.float32)
        
        block_actual_end_sample = min(current_block_start_sample + buffer_size, total_samples)
        actual_buffer_size_this_block = block_actual_end_sample - current_block_start_sample

        if actual_buffer_size_this_block != buffer_size:
            input_block_silence = input_block_silence[:, :actual_buffer_size_this_block]

        # Collect MIDI messages for this block
        block_midi_bytes = b''
        
        # Iterate through MIDI events that should start within this block
        temp_idx = midi_event_idx
        while temp_idx < len(midi_events):
            event_sample_offset, event_bytes = midi_events[temp_idx]
            if event_sample_offset < block_actual_end_sample : # Event falls within or starts before end of this block
                if event_sample_offset >= current_block_start_sample: # Event starts within this block
                    # The MIDI bytestring for pedalboard does not have explicit per-message timestamps.
                    # All messages in block_midi_bytes are effectively presented at the start of the block.
                    block_midi_bytes += event_bytes
                temp_idx +=1 # Check next event
            else: # Event is for a future block
                break 
        # midi_event_idx = temp_idx # Advance main index only if we consume messages for sure,
                                 # but for this simple concatenation, we only care about messages *starting* in block
                                 # This simple logic might send past messages again if not careful.
                                 # A better way:
        
        # Revised MIDI collection: only messages whose time falls into the current block window
        block_midi_bytes = b''
        while midi_event_idx < len(midi_events) and \
              midi_events[midi_event_idx][0] < block_actual_end_sample:
            # All MIDI messages whose nominal time is in this block are sent.
            # Their precise sub-block timing is lost to the plugin via this method;
            # they are effectively presented at the start of the block processing.
            block_midi_bytes += midi_events[midi_event_idx][1]
            midi_event_idx += 1
            
        # Process the block
        # The `midi` argument to `process` takes a bytestring of raw MIDI messages.
        processed_block = serum_instance.process(
            input_block_silence,
            sample_rate,
            midi=block_midi_bytes if block_midi_bytes else None
        )

        # Ensure processed_block has the correct shape (num_channels, actual_buffer_size_this_block)
        if processed_block.shape[1] != actual_buffer_size_this_block:
             # If plugin outputs fixed buffer size, truncate or pad if necessary
            processed_block = processed_block[:, :actual_buffer_size_this_block]

        output_audio[:, current_block_start_sample:block_actual_end_sample] = processed_block
        
        current_block_start_sample += actual_buffer_size_this_block
        
        # Progress indication
        progress = (i + 1) / num_blocks
        block_render_time = time.time() - block_start_time
        print(f"\rRendering block {i+1}/{num_blocks} ({progress*100:.1f}%) - "
              f"Block time: {block_render_time:.3f}s, "
              f"Est. remaining: {(num_blocks - (i+1)) * block_render_time:.1f}s", end="")

    print("\nRendering finished.")
    return output_audio


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Serum MIDI Renderer (via Pedalboard) ---")

    # 1. Load Serum Plugin
    serum = load_serum_plugin(SERUM_PLUGIN_PATH)
    if not serum:
        print("Exiting due to plugin load failure.")
        exit()

    # 2. TODO: Load and Apply Preset Parameters
    if PRESET_DATA_FILE_PATH:
        parsed_preset_parameters = load_parameters_from_file(PRESET_DATA_FILE_PATH)
        if parsed_preset_parameters:
            apply_parameters_to_serum(serum, parsed_preset_parameters)
        else:
            print("No preset parameters loaded or applied. Using Serum's current/default state.")
    else:
        print("No preset data file path specified. Using Serum's current/default state.")



    # 3. Parse MIDI File
    midi_events, total_render_samples = get_midi_events_and_total_samples(
        MIDI_INPUT_FILE_PATH, SAMPLE_RATE
    )
    if midi_events is None or total_render_samples == 0:
        print("Exiting due to MIDI processing failure.")
        exit()

    with open("serum_parameters.txt", "w") as f:
        f.write(str(serum.parameters))
    
    
    # 4. Render Audio
    rendered_audio = render_audio_from_midi(
        serum, midi_events, total_render_samples,
        SAMPLE_RATE, BUFFER_SIZE, OUTPUT_CHANNELS
    )

    
    # 5. Save Output Audio
    if rendered_audio is not None:
        try:
            # soundfile expects (frames, channels) but pedalboard uses (channels, frames)
            sf.write(AUDIO_OUTPUT_FILE_PATH, rendered_audio.T, int(SAMPLE_RATE))
            print(f"\nSuccessfully saved rendered audio to: {AUDIO_OUTPUT_FILE_PATH}")
        except Exception as e:
            print(f"\nError saving audio file: {e}")
    else:
        print("\nAudio rendering failed. No output file saved.")