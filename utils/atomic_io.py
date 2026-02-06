# utils/atomic_io.py
from __future__ import annotations
import os, json
from pathlib import Path
import pickle
import torch

def atomic_write_bytes(path: Path, data: bytes):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def atomic_write_json(path: Path, obj: dict):
    atomic_write_bytes(Path(path), json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

def atomic_pickle_dump(path: Path, obj):
    atomic_write_bytes(Path(path), pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

def atomic_torch_save(path: Path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    torch.save(obj, tmp)
    os.replace(tmp, path)