#!/bin/sh
# Usage: run_windows.sh NAME DATA_DIR DUMP_DIR JOBS FRAMES...
# Runs whole_scene.py over chunks of 8 frames, at most JOBS chunks at a time; logs in test_gaps/logs.
NAME=$1; DATA_DIR=$2; DUMP_DIR=$3; JOBS=$4; shift 4
LOGDIR="$HOME/Documents/MyArticles/dataset/test_gaps/logs"
mkdir -p "$LOGDIR"
cd "$HOME/Documents/MyArticles/code/3dseg" || exit 1
export V2X_SEQ_DIR="$DATA_DIR" SEG=mask MIRROR=0 TYRES=0 FIGURES=0 DUMP=1 DUMP_DIR="$DUMP_DIR" NAME LOGDIR
LIST=$(mktemp)
i=0
while [ $# -gt 0 ]; do
    chunk=""
    n=0
    while [ $# -gt 0 ] && [ $n -lt 8 ]; do chunk="$chunk $1"; shift; n=$((n + 1)); done
    echo "$i$chunk" >> "$LIST"
    i=$((i + 1))
done
xargs -P "$JOBS" -L 1 sh -c 'SUFFIX="_${NAME}_$0" python whole_scene.py "$@" > "$LOGDIR/${NAME}_$0.log" 2>&1' < "$LIST"
rm -f "$LIST"
echo "$NAME done: $(ls "docs/$DUMP_DIR" | wc -l) windows"
