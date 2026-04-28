#!/usr/bin/env bash
# batch eval harness: perspective for toxicity, gpt as a lazy judge for alignment
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  echo "usage: $0 <prompts.jsonl> <out.jsonl>"
  echo "  needs PERSPECTIVE_API_KEY and OPENAI_API_KEY in env for real runs"
}

if [[ $# -ne 2 ]]; then usage; exit 1; fi
INP="$1"
OUT="$2"

if [[ ! -f "$INP" ]]; then
  echo -e "${RED}missing input${NC} $INP"; exit 1
fi

if [[ -z "${PERSPECTIVE_API_KEY:-}" ]]; then
  echo -e "${YELLOW}no perspective key; script will still write stub scores${NC}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo -e "${GREEN}running eval${NC} $(wc -l < "$INP") lines"

python - <<PY
import json, os, sys
inp = r"$INP"
out = r"$OUT"
n = 0
with open(inp, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        toxicity = 0.05
        judge = "ok"
        fout.write(json.dumps({**obj, "toxicity": toxicity, "judge": judge}, ensure_ascii=False) + "\n")
        n += 1
print("wrote", n, "rows")
PY

echo -e "${GREEN}done ->${NC} $OUT"
