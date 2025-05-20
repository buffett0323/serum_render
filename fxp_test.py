import struct, zlib
from fxp import FXP
from pedalboard import load_plugin

SERUM_PLUGIN_PATH = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
PRESET_PATH = "serum preset/lead/LD - Starboy Era.fxp"
PLUGIN_NAME = "Serum 2" # "Serum 2 FX"


if __name__ == "__main__":
    # 1. Load preset
    fxp = FXP(PRESET_PATH)
    if not fxp.is_opaque():
        raise RuntimeError(f"{PRESET_PATH} is not a chunk‐based preset.")
    preset_chunk = fxp.data  # bytes
    
    # 2. Load serum plugin
    serum = load_plugin(SERUM_PLUGIN_PATH, plugin_name=PLUGIN_NAME)
    if serum is None:
        raise RuntimeError(f"Failed to load plugin at {SERUM_PLUGIN_PATH}")
    
    # 3. Apply the preset chunk to Serum’s state
    serum.preset_data = preset_chunk
    print(serum)
    # with open("serum_initial_state.txt", "w") as f:
    #     f.write(str(serum.preset_data))
        
    # with open(PRESET_PATH, "rb") as f:
    #     fxp_data = f.read()
    #     with open("serum_preset_data.txt", "w") as f:
    #         f.write(str(fxp_data))

    # # Parse FXP header (big-endian binary format)
    # chunkMagic, byteSize, fxMagic = struct.unpack(">4sI4s", fxp_data[:12])
    # pluginID_bytes = fxp_data[16:20]   # 4-byte plugin ID
    # plugin_id = pluginID_bytes.decode('latin1')
    # preset_name_bytes = fxp_data[28:56]  # 28-byte name field
    # preset_name = preset_name_bytes.split(b'\x00', 1)[0].decode('latin1')

    # print("Plugin ID:", plugin_id, "\n",
    #       "Preset Name:", preset_name)
    # # For Serum, plugin_id = "XfsX"

    # if fxMagic == b"FPCh":  # chunk-based preset
    #     # Read chunk size and chunk data
    #     chunk_size = struct.unpack(">I", fxp_data[56:60])[0]
    #     chunk_data = fxp_data[60:60 + chunk_size]
    #     # Decompress if the chunk is zlib-compressed
    #     try:
    #         chunk_data = zlib.decompress(chunk_data)
    #         print("Decompressed chunk size:", len(chunk_data))
    #     except zlib.error:
    #         print("Chunk is not compressed; size:", len(chunk_data))
    #     # chunk_data now holds Serum’s internal preset state (bytes)
    # else:
    #     # Parameter-list preset (not the case for Serum):
    #     param_count = struct.unpack(">I", fxp_data[24:28])[0]
    #     params = struct.unpack(">" + "f"*param_count, fxp_data[56:56 + 4*param_count])
    #     print(f"Loaded {param_count} parameter values.")
        
        