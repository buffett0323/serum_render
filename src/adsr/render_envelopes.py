import json
import random
from pathlib import Path
from util import GPT_ADSR


if __name__ == "__main__":
    OUT_DIR = Path("stats")
    OUT_DIR.mkdir(exist_ok=True)
    AMOUNT = 10000
    LENGTH_LIMIT = 2970
    DATA = GPT_ADSR
    
    meta = []
    stems = ["lead", "keys", "pad", "pluck", "synth", "vox"]
    
    ADSR_counter = 0
    MAX_len = 0
    for stem in stems:
        for i in range(AMOUNT):
            # Random parameters in milliseconds
            A = round(random.uniform(DATA[stem]["a1"], DATA[stem]["a2"]), 3)
            D = round(random.uniform(DATA[stem]["d1"], DATA[stem]["d2"]), 3)
            H = round(random.uniform(DATA[stem]["h1"], DATA[stem]["h2"]), 3)
            S = round(random.uniform(DATA[stem]["s1"], DATA[stem]["s2"]), 3)
            R = round(random.uniform(DATA[stem]["r1"], DATA[stem]["r2"]), 3)
            length = A + D + H + R
            
            if length >= LENGTH_LIMIT:
                # Adjust release to make total length < 2970
                R = LENGTH_LIMIT - (A + D + H)  # 2969 ensures length will be < 2970
                R = round(R, 3)
                length = A + D + H + R
                
            meta.append({
                "id": ADSR_counter,
                "stem": stem,
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

    with open(OUT_DIR / "envelopes_train.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)