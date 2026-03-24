#!/bin/bash

set -e

trap 'echo "FAIL"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_REPO_PATH="/home/emmanuel/Desktop/tesis/Visual_Place_Recognition"

if [ -z "$1" ]; then
    echo "Usage: $0 <num-seeds>"
    exit 1
fi
NUM_SEEDS="$1"

python "$SCRIPT_DIR/test_mapanything_salad_split.py" "$SCRIPT_DIR/samples/cimat_video/" $VPR_REPO_PATH --num-seeds "$NUM_SEEDS"


echo "PASS"
exit 0