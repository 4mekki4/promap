from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import PromptTrainDataset
from .training import build_formatter
from .modeling import load_model_checkpoint
from .utils import normalize_targets, p_at_1, resolve_device, softmax_if_needed


@dataclass
class RerankResult:
    merged_predictions: pd.DataFrame
    candidate_scores: pd.DataFrame
    final_predictions: pd.DataFrame
    metrics: dict[str, float]


def load_candidates(path: str | Path) -> pd.DataFrame:
    candidate_path = Path(path)
    if candidate_path.suffix in {".pickle", ".pkl"}:
        frame = pd.read_pickle(candidate_path)
    elif candidate_path.suffix == ".csv":
        frame = pd.read_csv(candidate_path)
    elif candidate_path.suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(candidate_path, sep="\t")
    else:
        raise ValueError(f"Unsupported candidate file format: {candidate_path.suffix}")

    required_columns = {"source", "candidate_top_k", "candidate_scores"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Candidate frame must contain the standardized columns "
            "['source', 'candidate_top_k', 'candidate_scores']. "
            f"Missing: {sorted(missing_columns)}."
        )

    for column in ("candidate_top_k", "candidate_scores"):
        if column in frame.columns:
            frame[column] = frame[column].apply(_parse_candidate_sequence)
    return frame


def _parse_candidate_sequence(value: object) -> list[object]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        raise ValueError(
            "Candidate list columns must be serialized as JSON arrays or Python-style lists."
        )
    if pd.isna(value):
        return []
    raise ValueError(f"Unsupported candidate list value: {value!r}")


def similarity_hit(candidates: Sequence[str], gold_targets: object) -> int:
    gold = set(normalize_targets(gold_targets))
    overlap = [candidate for candidate in candidates if candidate in gold]
    if not overlap:
        return -1
    return int(candidates[0] in gold)


def average_candidate_loss(token_losses: Sequence[float], target_subtokens: int) -> float:
    return float(np.mean(list(token_losses)[:target_subtokens]))


def compute_final_score(candidate_weight: float, token_loss: float) -> float:
    return float(candidate_weight * (1.0 / np.log1p(token_loss)))


def select_best_candidates(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    return candidate_scores.sort_values("final_score").drop_duplicates(subset="source", keep="last")


def normalize_candidate_frame(
    candidates: pd.DataFrame,
    *,
    temperature: float,
    top_k: int,
) -> pd.DataFrame:
    if not {"source", "candidate_top_k", "candidate_scores"}.issubset(candidates.columns):
        raise ValueError(
            "Candidate frame must contain the standardized columns "
            "['source', 'candidate_top_k', 'candidate_scores']."
        )
    normalized = candidates.loc[:, ["source", "candidate_top_k", "candidate_scores"]].copy()

    normalized["candidate_top_k"] = normalized["candidate_top_k"].apply(lambda row: list(row)[:top_k])
    normalized["candidate_scores"] = normalized["candidate_scores"].apply(
        lambda row: softmax_if_needed(list(row)[:top_k], temperature=temperature)
    )
    return normalized.drop_duplicates(subset="source", keep="first")


def merge_prompt_and_similarity(
    prompt_predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    temperature: float,
    top_k: int,
) -> pd.DataFrame:
    normalized_candidates = normalize_candidate_frame(
        candidates,
        temperature=temperature,
        top_k=top_k,
    )
    merged = prompt_predictions.merge(normalized_candidates, how="inner", on="source")
    merged["is_candidate_true"] = merged.apply(
        lambda row: similarity_hit(row["candidate_top_k"], row["target"]),
        axis=1,
    )
    return merged


def build_candidate_pairs(
    merged_predictions: pd.DataFrame,
    formatter,
    *,
    max_tokenized_word_length: int,
) -> pd.DataFrame:
    pairs: list[dict[str, object]] = []
    for row in merged_predictions.itertuples(index=False):
        source_length = formatter.word_length(row.source)
        for candidate, weight in zip(row.candidate_top_k, row.candidate_scores):
            target_length = formatter.word_length(candidate)
            if source_length > max_tokenized_word_length or target_length > max_tokenized_word_length:
                continue
            pairs.append(
                {
                    "source": row.source,
                    "target": candidate,
                    "target_subtokens": target_length,
                    "candidate_weight": weight,
                }
            )
    return pd.DataFrame(pairs)


def score_candidate_pairs(
    experiment: ExperimentConfig,
    checkpoint_path: str | Path,
    candidate_pairs: pd.DataFrame,
    *,
    device: str | None = None,
) -> pd.DataFrame:
    formatter = build_formatter(experiment)
    model = load_model_checkpoint(
        experiment.pretrained_path,
        formatter.tokenizer,
        checkpoint_path,
    ).to(resolve_device(device))
    model.eval()

    dataset = PromptTrainDataset(candidate_pairs, formatter, direction="s2t")
    data_loader = DataLoader(
        dataset,
        batch_size=experiment.batch_size,
        shuffle=False,
        num_workers=experiment.num_workers,
    )
    loss_criterion = nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=formatter.padding_token_id,
    )
    run_device = resolve_device(device)
    all_losses: list[list[float]] = []

    with torch.no_grad():
        for data_input, label_input in data_loader:
            for key, value in data_input.items():
                data_input[key] = value.to(run_device)
            targets = label_input["target"].to(run_device)
            logits = model(**data_input)
            token_losses = loss_criterion(logits, targets.view(-1))
            token_losses = token_losses.view(targets.shape[0], targets.shape[1])
            all_losses.extend(token_losses.cpu().tolist())

    scored = candidate_pairs.copy()
    scored["tokens_losses"] = all_losses
    scored["tokens_loss"] = scored.apply(
        lambda row: average_candidate_loss(row["tokens_losses"], int(row["target_subtokens"])),
        axis=1,
    )
    scored["final_score"] = scored.apply(
        lambda row: compute_final_score(float(row["candidate_weight"]), float(row["tokens_loss"])),
        axis=1,
    )
    return scored


def rerank_with_prompt_model(
    experiment: ExperimentConfig,
    checkpoint_path: str | Path,
    prompt_predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    device: str | None = None,
) -> RerankResult:
    formatter = build_formatter(experiment)
    merged = merge_prompt_and_similarity(
        prompt_predictions,
        candidates,
        temperature=experiment.similarity_temperature,
        top_k=experiment.similarity_top_k,
    )
    candidate_pairs = build_candidate_pairs(
        merged,
        formatter,
        max_tokenized_word_length=experiment.max_tokenized_word_length or experiment.num_prediction_tokens,
    )
    if candidate_pairs.empty:
        raise ValueError("No similarity candidates remain after token-length filtering.")

    scored_candidates = score_candidate_pairs(
        experiment,
        checkpoint_path,
        candidate_pairs,
        device=device,
    )
    best_candidates = select_best_candidates(scored_candidates).rename(
        columns={"target": "predicted_target"}
    )
    final_predictions = merged.merge(best_candidates, how="inner", on="source")
    final_predictions["is_prom_candidate_true"] = final_predictions.apply(
        lambda row: int(row["predicted_target"] in normalize_targets(row["target"])),
        axis=1,
    )
    metrics = {
        "prompt_p_at_1": p_at_1(final_predictions["is_true"]),
        "candidate_p_at_1": float((final_predictions["is_candidate_true"] == 1).mean()),
        "promap_p_at_1": p_at_1(final_predictions["is_prom_candidate_true"]),
    }
    return RerankResult(
        merged_predictions=merged,
        candidate_scores=scored_candidates,
        final_predictions=final_predictions,
        metrics=metrics,
    )
