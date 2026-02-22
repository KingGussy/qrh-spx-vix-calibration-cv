import json, time, os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path

# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)
PathLike = Union[str, os.PathLike, Path]
def _ensure_dir(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _log(msg: str) -> None:
    print(msg, flush=True)

def save_shard_npz(path: Path, rows: dict) -> None:
    np.savez_compressed(str(path), **rows)


# CV io utils:
def _now_tag():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


