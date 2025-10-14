"""Small list / aggregation helpers."""
from __future__ import annotations
from typing import List


def dedupe_list(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
