# Serum ADSR Dataset Generator

A Python-based tool for generating and analyzing ADSR (Attack, Decay, Sustain, Release) envelope datasets from Serum presets. This project helps in understanding and collecting ADSR parameter distributions across different sound categories in electronic music production.

## Features

- **ADSR Parameter Analysis**: Extracts and analyzes ADSR parameters from Serum presets
- **Multi-Category Support**: Processes presets across different sound categories:
  - Lead
  - Keys
  - Pad
  - Pluck
  - Synth
  - Vocals
- **Statistical Analysis**: Generates mean and standard deviation statistics for ADSR parameters
- **Visualization**: Includes tools for visualizing ADSR envelopes and waveforms

## Project Structure

```
.
├── src/
│   ├── adsr/
│   │   └── preset_adsr_stats.py    # Main script for ADSR parameter analysis
│   └── envelope_shaper_pb.py       # ADSR envelope visualization and generation
├── fxp_preset/
│   └── train/                      # Directory containing Serum presets
│       ├── lead/
│       ├── keys/
│       ├── pad/
│       └── ...
└── env_stats.json                  # Generated ADSR parameter statistics
```

## Requirements

- Python 3.8+
- dawdreamer
- numpy
- scipy
- tqdm
- Serum VST plugin

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure Serum VST is installed and update the `PLUGIN_PATH` in the scripts if necessary.

## Usage

### Analyzing ADSR Parameters

To analyze ADSR parameters from Serum presets:

```bash
python src/adsr/preset_adsr_stats.py
```

This will:
1. Scan all presets in the `fxp_preset/train` directory
2. Extract ADSR parameters from each preset
3. Generate statistics for each sound category
4. Save results to `env_stats.json`

### Generating ADSR Visualizations

To generate ADSR envelope visualizations:

```bash
python src/envelope_shaper_pb.py
```

This will:
1. Load audio samples
2. Apply ADSR envelopes with various parameters
3. Generate visualization plots showing:
   - ADSR envelope shape
   - Processed waveform
   - Original audio
4. Save audio files and metadata

## Output Format

### ADSR Statistics (env_stats.json)
```json
{
    "lead": {
        "Env1 Atk": {"mean": 0.5, "std": 0.2},
        "Env1 Dec": {"mean": 0.3, "std": 0.1},
        "Env1 Hold": {"mean": 0.2, "std": 0.1},
        "Env1 Sus": {"mean": 0.7, "std": 0.2},
        "Env1 Rel": {"mean": 0.4, "std": 0.2}
    },
    // ... other categories
}
```

### Generated Audio Files
- WAV files with applied ADSR envelopes
- PNG files showing envelope and waveform visualizations
- Metadata JSON containing ADSR parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- Serum by Xfer Records
- DawDreamer library