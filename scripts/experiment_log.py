from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = dict(row)
    row.setdefault("logged_at_utc", _utc_now_iso())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Stable column order: common keys first, then the rest (sorted).
    common = [
        "exp_id",
        "exp_name",
        "model_family",
        "model_name",
        "dataset",
        "n_samples",
        "split_train",
        "split_val",
        "split_test",
        "stage1_best_val_acc",
        "stage1_test_acc",
        "stage2_best_val_acc",
        "stage2_test_acc",
        "baseline_test_acc",
        "notes",
        "logged_at_utc",
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    rest = sorted(k for k in keys if k not in common)
    fieldnames = [k for k in common if k in keys] + rest

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_and_export(
    jsonl_path: str,
    csv_path: str,
    row: Dict[str, Any],
) -> None:
    append_jsonl(jsonl_path, row)
    rows = read_jsonl(jsonl_path)
    write_csv(csv_path, rows)

