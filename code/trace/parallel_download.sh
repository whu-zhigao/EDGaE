#!/bin/zsh
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: parallel_download.sh URL OUTPUT TOTAL_BYTES PARTS" >&2
  exit 2
fi

url=$1
output=$2
total=$3
parts=$4
part_dir="${output}.parts"
mkdir -p "$part_dir"

chunk=$(( (total + parts - 1) / parts ))
pids=()
for ((index=0; index<parts; index++)); do
  start=$((index * chunk))
  if (( start >= total )); then break; fi
  end=$((start + chunk - 1))
  if (( end >= total )); then end=$((total - 1)); fi
  part=$(printf '%s/part-%03d' "$part_dir" "$index")
  if [[ ! -s "$part" ]] || (( $(stat -f%z "$part") != end - start + 1 )); then
    curl -L --fail --silent --show-error --range "${start}-${end}" "$url" -o "$part" &
    pids+=($!)
  fi
done

for pid in $pids; do
  wait "$pid"
done

: > "$output"
for part in "$part_dir"/part-*(N); do
  cat "$part" >> "$output"
done
actual=$(stat -f%z "$output")
if (( actual != total )); then
  echo "assembled size mismatch: expected $total, got $actual" >&2
  exit 3
fi
echo "$output $actual"
