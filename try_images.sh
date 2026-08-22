#!/usr/bin/env bash
# Run the engine over your own photos.
#
# Drop images into my_images/ then run:
#
#   bash try_images.sh                 # process everything in my_images/
#   bash try_images.sh --conf 0.10     # lower the detection threshold
#   bash try_images.sh path/to/one.jpg # or just one image
#
# Annotated copies are written to my_images_out/.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$ROOT/cpp_engine/build/custom_engine"
IN_DIR="$ROOT/my_images"
OUT_DIR="$ROOT/my_images_out"

if [[ ! -x "$ENGINE" ]]; then
  echo "Engine not built. Run:" >&2
  echo "  cd $ROOT/cpp_engine && mkdir -p build && cd build && cmake .. && make" >&2
  exit 1
fi

# Anything that is not a flag is treated as a specific image to run. Flags are
# forwarded to the engine untouched.
IMAGES=()
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -*)
      PASSTHROUGH+=("$1")
      # Only consume a value if one is actually there and is not itself a flag.
      if [[ $# -ge 2 && "$2" != -* ]]; then
        PASSTHROUGH+=("$2")
        shift
      fi
      shift
      ;;
    *)
      IMAGES+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$IN_DIR" "$OUT_DIR"

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  while IFS= read -r f; do IMAGES+=("$f"); done < <(
    find "$IN_DIR" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | sort)
fi

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "No images found in $IN_DIR"
  echo "Copy some photos in there and run this again."
  exit 0
fi

echo "Running the engine over ${#IMAGES[@]} image(s). Roughly 4 seconds each."
echo

found_total=0
for img in "${IMAGES[@]}"; do
  base="$(basename "$img")"
  out="$OUT_DIR/${base%.*}_pred.jpg"

  echo "--- $base"
  # Expanding an empty array unquoted-safe: only forward flags if there are any,
  # otherwise an empty string reaches the engine and is parsed as an image path.
  if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
    "$ENGINE" "$img" -o "$out" "${PASSTHROUGH[@]}" 2>&1 | grep -Ev "^Loaded weights|^$"
  else
    "$ENGINE" "$img" -o "$out" 2>&1 | grep -Ev "^Loaded weights|^$"
  fi
  found_total=$((found_total + 1))
done

echo
echo "Annotated images are in $OUT_DIR"
echo "If nothing was detected, try a lower threshold, for example: bash try_images.sh --conf 0.10"
