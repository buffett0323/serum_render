import json


SAMPLE_RATE = 44100

with open("stats/serum_adsr.json", "r") as f:
    SERUM_ADSR = json.load(f)

with open("stats/gpt_adsr.json", "r") as f:
    GPT_ADSR = json.load(f)

def ms_to_samples(ms): return int(ms * SAMPLE_RATE / 1000)

def samples_to_ms(samples): return samples * 1000 / SAMPLE_RATE

def serum_to_ms(normalized_value, param_name):
    if "Sus" in param_name:  # Sustain is already in amplitude ratio
        return normalized_value
    
    return 32000 * (normalized_value ** 5) 

def convert_stats_to_ms(stats_path):
    stats_dict = json.load(open(stats_path))
    converted = {}
    for category in stats_dict:
        mean_in_ms = serum_to_ms(stats_dict[category]['mean'], category)
        std_in_ms = serum_to_ms(stats_dict[category]['std'], category)
        range_in_ms = (mean_in_ms - 2*std_in_ms, mean_in_ms + 2*std_in_ms)  # 95% confidence interval

        converted[category] = range_in_ms

    # Formatting
    final_converted = {"lead": {}, "keys": {}, "pad": {}, "pluck": {}, "synth": {}, "vox": {}}
    for name in converted:
        stem, item = name.split(', Env1 ')
        item = item[0].lower()
        
        if item == "s":
            c1 = max(0, converted[name][0])
            c2 = min(1, converted[name][1])
        else:
            c1 = max(0, converted[name][0])
            c2 = converted[name][1]
            
        final_converted[stem][f"{item}1"] = c1
        final_converted[stem][f"{item}2"] = c2
    return final_converted
