#!/usr/bin/env bash
# One-time setup. Needs internet; after this finishes, everything runs offline.
#
#   ./setup.sh                                  # CLIP ViT-B/32 (fast, ~600MB download)
#   ./setup.sh openai/clip-vit-large-patch14    # better matches, slower indexing
set -euo pipefail
cd "$(dirname "$0")"

HF_REPO="${1:-openai/clip-vit-base-patch32}"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# Apple's reference MLX CLIP implementation (model, tokenizer, image processor).
mkdir -p vendor/mlx_clip
BASE="https://raw.githubusercontent.com/ml-explore/mlx-examples/main/clip"
for f in clip.py model.py tokenizer.py image_processor.py convert.py requirements.txt; do
  curl -fsSL "$BASE/$f" -o "vendor/mlx_clip/$f"
done

pip install -r requirements.txt -r vendor/mlx_clip/requirements.txt

# Download the Hugging Face weights and convert them to MLX format.
python vendor/mlx_clip/convert.py --hf-repo "$HF_REPO" --mlx-path mlx_model

echo
echo "Done. Try:"
echo "  source .venv/bin/activate"
echo "  python -m framefinder index ~/Movies/my_videos"
echo "  python -m framefinder search \"a red car driving at night\""
