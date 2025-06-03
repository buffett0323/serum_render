python ddw_render_chain.py \
    --plugin /Users/bliu/Library/Audio/Plug-Ins/VST/Serum.vst \
    --preset-dir ../fxp_preset/lead \
    --sample-rate 44100 \
    --bpm 120 \
    --render-duration 10 \
    --num-workers 4 \
    --output-dir ../rendered_audio/lead \
    --log-level INFO \
    --split evaluation
