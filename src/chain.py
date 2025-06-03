import soundfile as sf
from pedalboard import Pedalboard, Compressor, Delay, Reverb, load_plugin
from pedalboard.io import AudioFile

# 1. Read in a whole file, resampling to our desired sample rate:
samplerate = 44100.0
with AudioFile('demo.wav').resampled_to(samplerate) as f:
    audio = f.read(f.frames)
  

# 2. 掛載第三方 VST3 / AU
eq          = load_plugin("/Library/Audio/Plug-Ins/VST3/FabFilter Pro-Q 3.vst3")   # EQ
saturator   = load_plugin("/Library/Audio/Plug-Ins/VST3/FabFilter Saturn 2.vst3")     # 飽和
# imager      = load_plugin("/Library/Audio/Plug-Ins/VST3/Ozone 9 Imager.vst3")      # 立體寬度
ott         = load_plugin("/Library/Audio/Plug-Ins/Components/OTT.component")                 # 多段壓縮
# sidechain   = load_plugin("/Library/Audio/Plug-Ins/VST/Kickstart.vst")
# compressor   = load_plugin("/Library/Audio/Plug-Ins/VST3/FabFilter Pro-C 2.vst3")         # Compressor
# reverb       = load_plugin("/Library/Audio/Plug-Ins/VST3/FabFilter Pro-R.vst3")         # Reverb

# 3. Pedalboard 內建效果
compressor = Compressor(threshold_db=-12, ratio=4, attack_ms=5, release_ms=50)
delay      = Delay(delay_seconds=0.25, feedback=0.35, mix=0.25)   # 1/4 拍延遲
reverb     = Reverb(room_size=0.5, wet_level=0.2)

# 4. 排列效果鏈（Serum ➜ EQ ➜ Saturation …）
board = Pedalboard([
    eq,
    saturator,
    compressor,
    # imager,
    delay,
    reverb,
    ott,
    # sidechain,
])  # sample_rate 參數自 Pedalboard v0.9 起可直接傳入 [oai_citation:8‡spotify.github.io](https://spotify.github.io/pedalboard/examples.html?utm_source=chatgpt.com)

# 5. 處理並輸出
processed = board(audio, sample_rate=samplerate)

# Write the audio back as a wav file:
with AudioFile('processed-demo.wav', 'w', samplerate, processed.shape[0]) as f:
  f.write(processed)