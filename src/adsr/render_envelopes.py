import json
import random
from pathlib import Path


if __name__ == "__main__":
    OUT_DIR = Path("stats")
    OUT_DIR.mkdir(exist_ok=True)
    AMOUNT = 100
    LENGTH_LIMIT = 3000
    DATA = {
        "ATTACK": [0, 500],
        "DECAY": [100, 1000],
        "SUSTAIN": [0, 1],
        "RELEASE": [100, 1500],
        "HOLD": [0, 1],
    }
    
    meta = []

    ADSR_counter = 0
    MAX_len = 0
    for i in range(AMOUNT):
        # Random parameters in milliseconds
        A = round(random.uniform(DATA["ATTACK"][0], DATA["ATTACK"][1]), 3)
        D = round(random.uniform(DATA["DECAY"][0], DATA["DECAY"][1]), 3)
        H = round(random.uniform(DATA["HOLD"][0], DATA["HOLD"][1]), 3)
        S = round(random.uniform(DATA["SUSTAIN"][0], DATA["SUSTAIN"][1]), 3)
        R = round(random.uniform(DATA["RELEASE"][0], DATA["RELEASE"][1]), 3)
        length = A + D + H + R
        
        if length >= LENGTH_LIMIT:
            # Adjust release to make total length < 2970
            R = LENGTH_LIMIT - (A + D + H)  # 2969 ensures length will be < 2970
            R = round(R, 3)
            length = A + D + H + R
            
        meta.append({
            "id": ADSR_counter,
            "attack": A,
            "decay": D,
            "hold": H,
            "sustain": S,
            "release": R,
            "length": length
        })
        ADSR_counter += 1
        MAX_len = max(MAX_len, A+D+H+R)
    
    print("MAX_LENGTH:", MAX_len)

    with open(OUT_DIR / "envelopes_train_new.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)