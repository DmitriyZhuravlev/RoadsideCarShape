#!/bin/sh
# Pedestrians, cyclists, motorcyclists of V2X-Seq-SPD near the camera (own tracking, no labels but the selection),
# one at a time; results in 3dseg/docs/vru_spd/<sequence>/.
cd "$HOME/Documents/MyArticles/code/3dseg" || exit 1
for st in seq_0077:000118 seq_0059:003926 seq_0080:000081 seq_0036:002251 seq_0014:003579 seq_0093:000911 \
          seq_0063:003765 seq_0008:001365 seq_0084:000146; do
    seq=${st%%:*}; tid=${st##*:}
    V2X_SEQ_DIR="$HOME/Documents/MyArticles/dataset/spd_seq/$seq" VRU_OUT="docs/vru_spd/$seq" \
        nice -n 15 python vru_shape.py "$tid" 2>&1 | grep -v "^ground" | tail -1 | sed "s/^/$seq /"
done
