#!/usr/bin/env bash
# Download the KITTI 2D object detection split used by the AdverTest benchmark.
#
#   bash scripts/fetch_kitti.sh --subset 500     # MVP subset of plan §1.1
#   bash scripts/fetch_kitti.sh                  # full 7481-frame training split
#
# KITTI is published by Geiger, Lenz & Urtasun (CVPR 2012) under CC BY-NC-SA 3.0
# for non-commercial research. The landing page, which carries the licence and the
# citation you owe them, is:
#   https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
#
# Note: KITTI frames are NOT anonymised. src/datasets/kitti.py runs a PLACEHOLDER
# anonymiser over anything it loads; that is enough to open the pipeline's §6 gate
# and nothing more. Do not redistribute these images.
set -euo pipefail

ROOT="${ADVERTEST_KITTI_ROOT:-data/kitti}"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti"
SPLIT_URL="https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets"
SUBSET=""
ASSUME_YES=0
KEEP_TESTING=0

usage() {
    sed -n '2,14p' "$0"
    cat <<'EOF'

Options:
  --root PATH     where to unpack (default: $ADVERTEST_KITTI_ROOT or data/kitti)
  --subset N      keep only the first N frames of the val split (deletes the rest)
  --keep-testing  keep testing/ as well (unlabelled, ~6 GB, unused by the benchmark)
  --yes           do not ask before deleting frames outside the subset
  --help          this text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --subset) SUBSET="$2"; shift 2 ;;
        --keep-testing) KEEP_TESTING=1; shift ;;
        --yes) ASSUME_YES=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

for tool in curl unzip; do
    command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

mkdir -p "$ROOT" "$ROOT/ImageSets"
cd "$ROOT"
echo "==> KITTI root: $(pwd)"

fetch() {
    local name="$1"
    if [[ -f "$name" ]]; then
        echo "==> $name already downloaded, resuming if incomplete"
    fi
    # -C - resumes a partial file; the image archive is ~12 GB.
    curl -fL -C - --retry 3 -o "$name" "$BASE_URL/$name"
}

echo "==> downloading labels (~5 MB)"
fetch data_object_label_2.zip
echo "==> downloading left colour images (~12 GB, resumable)"
fetch data_object_image_2.zip

echo "==> unpacking"
unzip -q -o data_object_label_2.zip
unzip -q -o data_object_image_2.zip

if [[ $KEEP_TESTING -eq 0 && -d testing ]]; then
    echo "==> removing testing/ (unlabelled, not used by the benchmark)"
    rm -rf testing
fi

echo "==> fetching the Chen (3DOP) train/val split ids"
for split in train val; do
    curl -fsSL -o "ImageSets/${split}.txt" "$SPLIT_URL/${split}.txt"
done

n_images=$(find training/image_2 -name '*.png' | wc -l)
n_labels=$(find training/label_2 -name '*.txt' | wc -l)
echo "==> unpacked $n_images images / $n_labels label files"
if [[ "$n_images" -ne "$n_labels" ]]; then
    echo "!! image and label counts differ — the download is incomplete" >&2
    exit 1
fi

if [[ -n "$SUBSET" ]]; then
    keep_list=$(mktemp)
    head -n "$SUBSET" ImageSets/val.txt > "$keep_list"
    n_keep=$(wc -l < "$keep_list")
    n_delete=$((n_images - n_keep))
    echo "==> subset mode: keeping $n_keep frames, deleting $n_delete"
    if [[ $ASSUME_YES -eq 0 ]]; then
        read -r -p "    delete $n_delete images and labels under $(pwd)/training? [y/N] " reply
        [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; rm -f "$keep_list"; exit 1; }
    fi
    while read -r image; do
        id=$(basename "$image" .png)
        grep -qxF "$id" "$keep_list" || rm -f "training/image_2/$id.png" "training/label_2/$id.txt"
    done < <(find training/image_2 -name '*.png')
    cp "$keep_list" ImageSets/val.txt
    : > ImageSets/train.txt
    rm -f "$keep_list"
    echo "==> subset kept: $(find training/image_2 -name '*.png' | wc -l) frames"
fi

cat <<EOF

Done. Point the loader at it with:

    export ADVERTEST_KITTI_ROOT=$(pwd)
    python -m src.cli datasets

Then run the benchmark:

    python scripts/benchmark_kitti_yolo11.py --limit 500
EOF
