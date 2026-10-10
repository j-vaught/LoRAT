#!/usr/bin/env bash
set -euo pipefail
url='http://127.0.0.1:8876'
server='comech-1176'
if ! curl -fsS "$url/api/config" | python3 -c 'import json,sys; assert json.load(sys.stdin)["model"] == "LoRAT g-378"' >/dev/null 2>&1; then
  ssh "$server" 'app=$(find "$HOME/Documents" -path "*/hi/work/LoRAT/browser_app/start.sh" -print -quit); test -n "$app" && bash "$app"'
  ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:8876:127.0.0.1:8766 "$server"
fi
open "$url"
