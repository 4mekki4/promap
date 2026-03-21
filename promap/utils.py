from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import random
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def maybe_resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def normalize_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def prediction_head(text: Any) -> str:
    normalized = " ".join(str(text).split())
    return normalized.split()[0] if normalized else ""


def softmax_if_needed(weights: Sequence[float], temperature: float = 1.0) -> list[float]:
    values = np.asarray(list(weights), dtype=float)
    if values.size == 0:
        return []
    if np.all(values >= 0) and np.isclose(values.sum(), 1.0, atol=1e-3):
        return values.tolist()
    temp = temperature if temperature > 0 else 1.0
    shifted = values / temp
    shifted -= shifted.max()
    exp_values = np.exp(shifted)
    denom = exp_values.sum()
    if denom <= 0:
        return np.full_like(exp_values, 1.0 / len(exp_values)).tolist()
    return (exp_values / denom).tolist()


def p_at_1(flags: Iterable[int | bool]) -> float:
    values = [int(bool(flag)) for flag in flags]
    return float(np.mean(values)) if values else 0.0
