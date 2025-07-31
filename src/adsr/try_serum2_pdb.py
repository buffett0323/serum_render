from pedalboard import (
    Pedalboard, Compressor, Delay, Reverb, load_plugin,
)
import os

preset_path = "../../fxp_preset/serum2/Lead/LD - Analog Glow.SerumPreset"
serum2 = load_plugin("/Library/Audio/Plug-Ins/VST3/Serum2.vst3", plugin_name="Serum 2")
print(type(serum2))
print(os.path.exists(preset_path))

serum2.load_preset(preset_path)





