#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(dirname "$app_dir")"
runtime_dir="$repo_dir/work/browser"
mkdir -p "$runtime_dir"
if curl -fsS http://127.0.0.1:8766/api/config >/dev/null 2>&1; then
  echo 'LoRAT browser is already running on port 8766.'
  exit 0
fi
cd "$repo_dir"
source_args=()
if [[ -f "$runtime_dir/source.json" ]]; then
  source_args=(--source-config "$runtime_dir/source.json")
fi
nohup "$repo_dir/.venv/bin/python" "$app_dir/server.py" --device "${LORAT_DEVICE:-cuda:1}" "${source_args[@]}" > "$runtime_dir/server.log" 2>&1 < /dev/null &
echo "$!" > "$runtime_dir/server.pid"
for attempt in {1..30}; do
  if curl -fsS http://127.0.0.1:8766/api/config >/dev/null 2>&1; then
    echo 'LoRAT browser is ready on port 8766.'
    exit 0
  fi
  sleep 1
done
cat "$runtime_dir/server.log"
exit 1
