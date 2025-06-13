import numpy as np, soundfile as sf, json, random, uuid
from pathlib import Path
from tqdm import trange, tqdm
from util import SERUM_ADSR





if __name__ == "__main__":
    OUT_DIR = Path("stats")
    OUT_DIR.mkdir(exist_ok=True)
    
    meta = []
    stems = ["lead", "keys", "pad", "pluck", "synth", "vox"]
    
    ADSR_counter = 0
    
    for stem in stems:
        for i in trange(100):
            # Random parameters in milliseconds
            A = round(random.uniform(SERUM_ADSR[stem]["a1"], SERUM_ADSR[stem]["a2"]), 3)
            D = round(random.uniform(SERUM_ADSR[stem]["d1"], SERUM_ADSR[stem]["d2"]), 3)
            H = round(random.uniform(SERUM_ADSR[stem]["h1"], SERUM_ADSR[stem]["h2"]), 3)
            S = round(random.uniform(SERUM_ADSR[stem]["s1"], SERUM_ADSR[stem]["s2"]), 3)
            R = round(random.uniform(SERUM_ADSR[stem]["r1"], SERUM_ADSR[stem]["r2"]), 3)
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
        

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)