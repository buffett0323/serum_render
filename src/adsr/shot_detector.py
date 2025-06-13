import os
import numpy as np
import librosa
from pathlib import Path
from glob import glob

# ===== 可自行調整的參數 =====
IN_DIR  = "../../rendered_one_shot"    # 原始音檔資料夾
OUT_DIR = "sustained"     # 篩選後輸出目錄
SR          = 44100       # 重採樣取樣率
PREFIX      = 0.5
WINDOW_SEC  = 1.5         # 檢查的長度（秒）
FRAME_LEN   = 2048        # RMS frame length（samples）
HOP_LEN     = 512         # RMS hop length（samples）
MIN_RMS     = 0.02        # 平均 RMS 要大於此值才算有訊號
CV_THRESH   = 0.30        # RMS 變異係數 <= 0.30 視為能量「夠平穩」
# SLOPE_THRESH = -0.001     # log-RMS 隨時間斜率要 >= -0.001

# ==================================
os.makedirs(OUT_DIR, exist_ok=True)

def is_sustained(path: str) -> bool:
    """回傳 True 表示前 1.5 秒能量高且平穩"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < WINDOW_SEC * SR:
        return False
    
    y = y[int(PREFIX * SR): int((WINDOW_SEC+PREFIX) * SR)]
    rms = librosa.feature.rms(
        y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN, center=False
    )[0]
    
    mean_rms = rms.mean()
    if mean_rms < MIN_RMS: return False # Almost silent
        
    
    cv_rms = rms.std() / (mean_rms + 1e-9)    # 變異係數
    
    # 也可量化能量衰減斜率（對 RMS 取 log 後線性回歸）
    # times   = np.arange(len(rms)) * HOP_LEN / SR
    # slope, _ = np.polyfit(times, np.log(rms + 1e-12), 1)
    

    return (cv_rms <= CV_THRESH)# and (slope >= SLOPE_THRESH)



if __name__ == "__main__":
    kept, dropped = 0, 0
    kept_paths = []
    for fname in os.listdir(IN_DIR):
        if not fname.lower().endswith((".wav", ".flac", ".aiff", ".aif", ".ogg")):
            continue
        
        fpath = os.path.join(IN_DIR, fname)
        try:
            if is_sustained(fpath):
                kept_paths.append(fname)
                kept += 1
            else:
                dropped += 1
        except Exception as e:
            print(f"[Warn] {fname}: {e}")

    print(f"Kept {kept} files, dropped {dropped}.") # Kept 1547 files, dropped 1747.

    with open("stats/chosen_timbre_content_pairs.txt", "w") as f:
        for path in kept_paths:
            f.write(path + "\n")