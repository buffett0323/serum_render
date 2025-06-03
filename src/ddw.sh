python ddw_midis.py \
    --plugin /Library/Audio/Plug-Ins/VST/Serum.vst \
    --preset-dir ../fxp_preset/lead_out \
    --sample-rate 44100 \
    --bpm 120 \
    --render-duration 10 \
    --num-workers 4 \
    --output-dir ../rendered_audio/lead_out \
    --log-level INFO