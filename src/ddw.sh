python ddw_midis.py \
    --plugin /Library/Audio/Plug-Ins/VST/Serum.vst \
    --preset-dir ../fxp_preset/lead \
    --sample-rate 44100 \
    --bpm 120 \
    --render-duration 10 \
    --num-workers 1 \
    --output-dir ../rendered_audio/lead \
    --log-level INFO