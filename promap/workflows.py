from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, load_experiment_config
from .reranking import load_candidates, rerank_with_prompt_model
from .results import summarize_arabic_pickles, summarize_promap_pickles
from .training import train_prompt_model
from .utils import ensure_dir


def select_pairs(
    config: ExperimentConfig,
    requested_pairs: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if not requested_pairs:
        return config.pairs
    valid = set(config.pairs)
    missing = [pair for pair in requested_pairs if pair not in valid]
    if missing:
        missing_text = ", ".join(f"{src}_{tgt}" for src, tgt in missing)
        raise ValueError(f"Requested pairs are not in the config: {missing_text}")
    return requested_pairs


def build_candidate_path(
    config: ExperimentConfig,
    pair: tuple[str, str],
    override_dir: str | None,
    override_template: str | None,
) -> Path | None:
    candidate_dir = Path(override_dir).resolve() if override_dir else config.similarity_candidates_dir
    candidate_template = override_template or config.similarity_path_template
    if not candidate_dir or not candidate_template:
        return None
    return candidate_dir / candidate_template.format(src=pair[0], tgt=pair[1])


def run_scenario(
    config_path: str | Path,
    *,
    pairs: list[tuple[str, str]] | None = None,
    device: str | None = None,
    skip_rerank: bool = False,
    candidate_dir: str | None = None,
    candidate_template: str | None = None,
) -> None:
    config = load_experiment_config(config_path)
    selected = select_pairs(config, pairs)

    checkpoint_dir = ensure_dir(config.output_dir / config.checkpoint_dir)
    prompt_dir = ensure_dir(config.output_dir / config.prompt_predictions_dir)
    rerank_dir = ensure_dir(config.output_dir / config.reranked_predictions_dir)
    extra_dir = ensure_dir(rerank_dir / "extra")

    for pair in selected:
        pair_name = config.pair_name(pair)
        checkpoint_path = checkpoint_dir / f"{pair_name}.pth"
        prompt_path = prompt_dir / f"{pair_name}_prompt.pkl"
        metrics_path = prompt_dir / f"{pair_name}_prompt_metrics.json"

        result = train_prompt_model(
            config,
            pair,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        result.prompt_predictions.to_pickle(prompt_path)
        metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        print(f"[prompt] {pair_name}: P@1={result.metrics['prompt_p_at_1']:.4f}")

        if skip_rerank:
            continue

        resolved_candidate_path = build_candidate_path(
            config,
            pair,
            candidate_dir,
            candidate_template,
        )
        if resolved_candidate_path is None or not resolved_candidate_path.exists():
            print(f"[rerank] {pair_name}: skipped, no candidate file found.")
            continue

        rerank_result = rerank_with_prompt_model(
            config,
            checkpoint_path,
            result.prompt_predictions,
            load_candidates(resolved_candidate_path),
            device=device,
        )
        final_path = rerank_dir / f"{pair_name}_preds.pickle"
        rerank_result.final_predictions.to_pickle(final_path)
        (rerank_dir / f"{pair_name}_metrics.json").write_text(
            json.dumps(rerank_result.metrics, indent=2),
            encoding="utf-8",
        )
        if config.store_intermediate:
            rerank_result.merged_predictions.to_pickle(extra_dir / f"{pair_name}_before_merge.pickle")
            rerank_result.candidate_scores.to_pickle(extra_dir / f"{pair_name}_all_preds.pickle")
        print(f"[rerank] {pair_name}: P@1={rerank_result.metrics['promap_p_at_1']:.4f}")


def rerank_saved_predictions(
    config_path: str | Path,
    *,
    checkpoint: str | Path,
    prompt_predictions: str | Path,
    candidates: str | Path,
    output: str | Path,
    metrics_output: str | Path | None = None,
    device: str | None = None,
) -> dict[str, float]:
    config = load_experiment_config(config_path)
    rerank_result = rerank_with_prompt_model(
        config,
        checkpoint,
        pd.read_pickle(prompt_predictions),
        load_candidates(candidates),
        device=device,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rerank_result.final_predictions.to_pickle(output_path)
    if metrics_output:
        Path(metrics_output).write_text(
            json.dumps(rerank_result.metrics, indent=2),
            encoding="utf-8",
        )
    return rerank_result.metrics


def summarize_local_results(
    preds_dir: str | Path,
    arabic_preds_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
):
    promap_summary = summarize_promap_pickles(preds_dir)
    arabic_summary = summarize_arabic_pickles(arabic_preds_dir)

    if output_dir:
        output_root = ensure_dir(output_dir)
        promap_summary.to_csv(output_root / "promap_summary.csv", index=False)
        arabic_summary.to_csv(output_root / "arabic_summary.csv", index=False)
    return promap_summary, arabic_summary
