from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data import prompt_prediction_is_correct
from .utils import normalize_targets

PROMAP_FILE_RE = re.compile(r"(?P<src>[^_]+)_(?P<tgt>[^_]+)_(?P<size>\d+)_preds\.pickle$")
ARABIC_FILE_RE = re.compile(r"(?P<dialect>.+)_msa_(?P<direction>s2t|t2s)\.pickle$")


def _resolve_gold_column(frame: pd.DataFrame) -> str:
    for column in ("target", "targets", "target_x"):
        if column in frame.columns:
            return column
    raise ValueError("Could not find the gold target column.")


def _resolve_reranked_column(frame: pd.DataFrame) -> str:
    for column in ("predicted_target", "target_y"):
        if column in frame.columns:
            return column
    raise ValueError("Could not find the reranked prediction column.")


def summarize_promap_pickles(pred_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(pred_dir).glob("*_preds.pickle")):
        match = PROMAP_FILE_RE.match(path.name)
        frame = pd.read_pickle(path)
        rows.append(
            {
                "file": path.name,
                "source_lang": match.group("src") if match else None,
                "target_lang": match.group("tgt") if match else None,
                "train_size": int(match.group("size")) if match else None,
                "examples": len(frame),
                "prompt_p_at_1": float(frame["is_true"].mean()) if "is_true" in frame else None,
                "candidate_p_at_1": float((frame["is_candidate_true"] == 1).mean())
                if "is_candidate_true" in frame
                else None,
                "promap_p_at_1": float(frame["is_prom_candidate_true"].mean())
                if "is_prom_candidate_true" in frame
                else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_arabic_pickles(pred_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(pred_dir).glob("*.pickle")):
        frame = pd.read_pickle(path)
        gold_column = _resolve_gold_column(frame)
        top_k_available = "s2t_top_k" in frame.columns
        prompt_hits = frame.apply(
            lambda row: prompt_prediction_is_correct(row["s2t_pred"], row[gold_column]),
            axis=1,
        )
        if top_k_available:
            top_10_hits = frame.apply(
                lambda row: int(bool(set(normalize_targets(row["s2t_top_k"])) & set(normalize_targets(row[gold_column])))),
                axis=1,
            )
        else:
            top_10_hits = None

        match = ARABIC_FILE_RE.match(path.name)
        rows.append(
            {
                "file": path.name,
                "dialect": match.group("dialect") if match else None,
                "direction": match.group("direction") if match else None,
                "examples": len(frame),
                "p_at_1": float(prompt_hits.mean()),
                "p_at_10": float(top_10_hits.mean()) if top_10_hits is not None else None,
            }
        )
    return pd.DataFrame(rows)
