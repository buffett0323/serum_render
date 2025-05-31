import dawdreamer as daw
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE = 44100
BLOCK_SIZE  = 512
PLUGIN_PATH = "/Library/Audio/Plug-Ins/VST3/FabFilter Pro-Q 3.vst3"  # ←改成你的路徑
INPUT_WAV   = "input.wav"
OUTPUT_WAV  = "output_with_eq.wav"

# 1) 啟動 DSP 引擎
engine = daw.Engine(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)

# 2) 建立 Processor
player = engine.make_buffer_reader("player", INPUT_WAV, loop=False)
eq     = engine.make_plugin_processor("fabfilter_eq", PLUGIN_PATH)
writer = engine.make_buffer_writer("writer", OUTPUT_WAV)

# ★(可選) 讀取並列印所有可調參數名稱，方便之後自動化
print(eq.get_parameter_names())        # 需要 DawDreamer ≥ 0.6.3

# ★(可選) 直接用索引或名稱設定參數；下面示範把第一個 Band 設成
#    100 Hz, Gain +3 dB, Q 1.0（不同版本索引可能不同，建議先列印名稱確認）
eq.set_parameter_by_name("Band1 Frequency", 100.0)
eq.set_parameter_by_name("Band1 Gain",       3.0)
eq.set_parameter_by_name("Band1 Q",          1.0)

# 3) 建立訊號流向 (Directed Acyclic Graph)
graph = [
    (player, []),             # player 無輸入
    (eq,     [player.get_name()]),
    (writer, [eq.get_name()])
]
engine.load_graph(graph)

# 4) 依音檔長度離線渲染
n_frames = int(sf.info(INPUT_WAV).frames)
engine.render(n_frames)

print("Done! File saved at", os.path.abspath(OUTPUT_WAV))