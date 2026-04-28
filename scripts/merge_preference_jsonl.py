# merge jsonl preference shards from turkers; dedupe on prompt id best effort
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple


def stable_key(obj: Dict[str, Any]) -> str:
    base = json.dumps(
        {"p": obj.get("prompt"), "a": obj.get("a"), "b": obj.get("b")},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def merge(paths: Iterable[Path]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        for obj in iter_jsonl(p):
            k = stable_key(obj)
            if k in seen:
                continue
            seen[k] = obj
            yield k, obj


def write_merged(out: Path, paths: Iterable[Path]) -> int:
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for _, obj in merge(paths):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    import sys

    files = [Path(x) for x in sys.argv[1:]]
    print("rows", write_merged(Path("merged_prefs.jsonl"), files))
