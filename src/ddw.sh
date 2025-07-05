python ddw_midis.py \
    --plugin /Library/Audio/Plug-Ins/VST/Serum.vst \
    --preset-dir ../fxp_preset/train \
    --sample-rate 44100 \
    --bpm 120 \
    --render-duration 3 \
    --num-workers 4 \
    --output-dir ../rendered_audio/train \
    --log-level INFO \
    --split train