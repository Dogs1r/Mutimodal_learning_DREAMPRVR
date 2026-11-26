#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
: "${BASE:=$repo/work}"
: "${CHECKPOINT:?Set CHECKPOINT to your Charades DreamPRVR backbone checkpoint}"
text="$BASE/data/charades/TextData"
visual="$BASE/data/charades/FeatureData/i3d_rgb_lgi"
queries=${QUERY_TEXT:-"$BASE/runs/queries.h5"}
if [[ ! -f "$queries" ]]; then
    python "$repo/src/encode_clip_queries.py" --weights "$BASE/weights/ViT-B-32.pt" \
        --tokenizer "$BASE/weights/clip-tokenizer/tokenizer.json" --output "$queries" \
        "$text/charadestrain.caption.txt" "$text/charadestest.caption.txt"
fi
python "$repo/src/native_clip.py" test --two-stage --checkpoint "$CHECKPOINT" \
    --data-root "$BASE/data" --train "$text/charadestrain.caption.txt" \
    --eval "$text/charadestest.caption.txt" --visual "$visual/new_clip_vit_32_charades_vid_features.hdf5" \
    --train-text "$text/clip_ViT_B_32_charades_query_feat.hdf5" --query-text "$queries" \
    --output "${OUTPUT:-$BASE/runs/two-stage/test}"
