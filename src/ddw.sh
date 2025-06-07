python ddw_midis.py \
    --plugin /Users/bliu/Library/Audio/Plug-Ins/VST/Serum.vst \
    --preset-dir ../fxp_preset/train \
    --sample-rate 44100 \
    --bpm 120 \
    --render-duration 5 \
    --num-workers 4 \
    --output-dir ../rendered_audio/train \
    --log-level INFO \
    --split train