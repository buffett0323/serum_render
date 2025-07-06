import json
import random
from pathlib import Path


if __name__ == "__main__":
    OUT_DIR = Path("stats")
    OUT_DIR.mkdir(exist_ok=True)
    AMOUNT = 50
    LENGTH_LIMIT = 3000
    DATA = {
        "pluck": {
            "ATTACK":  (10, 50),
            "DECAY":   (50, 200),
            "HOLD":    (0, 20),
            "SUSTAIN": (0.0, 0.20),
            "RELEASE": (30, 120),
        },
        "lead": {
            "ATTACK":  (10, 100),
            "DECAY":   (100, 300),
            "HOLD":    (0, 200),
            "SUSTAIN": (0.40, 0.80),
            "RELEASE": (80, 300),
        },
    }
    
    meta = []

    ADSR_counter = 0
    MAX_len = 0
    for stem in DATA.keys():
        for i in range(AMOUNT):
            # Random parameters in milliseconds
            A = round(random.uniform(DATA[stem]["ATTACK"][0], DATA[stem]["ATTACK"][1]), 3)
            D = round(random.uniform(DATA[stem]["DECAY"][0], DATA[stem]["DECAY"][1]), 3)
            H = round(random.uniform(DATA[stem]["HOLD"][0], DATA[stem]["HOLD"][1]), 3)
            S = round(random.uniform(DATA[stem]["SUSTAIN"][0], DATA[stem]["SUSTAIN"][1]), 3)
            R = round(random.uniform(DATA[stem]["RELEASE"][0], DATA[stem]["RELEASE"][1]), 3)
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
                "length": length,
                "stem": stem
            })
            ADSR_counter += 1
            MAX_len = max(MAX_len, A+D+H+R)
    
    print("MAX_LENGTH:", MAX_len)
    random.shuffle(meta)

    with open(OUT_DIR / "envelopes_train_new.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)