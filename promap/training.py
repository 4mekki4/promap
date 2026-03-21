from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig, Pair
from .data import PromptFormatter, build_dataloaders, build_prompt_prediction_frame, read_dictionary
from .modeling import load_model, load_tokenizer
from .utils import p_at_1, resolve_device, seed_everything


@dataclass
class PromptRunResult:
    train_pairs: pd.DataFrame
    test_pairs: pd.DataFrame
    prompt_predictions: pd.DataFrame
    metrics: dict[str, float]
    checkpoint_path: Path | None = None


def build_formatter(experiment: ExperimentConfig) -> PromptFormatter:
    tokenizer = load_tokenizer(
        experiment.pretrained_path,
        [
            experiment.prediction_token,
            experiment.source_pad_token or experiment.prediction_token,
            experiment.padding_token,
        ],
    )
    return PromptFormatter(
        tokenizer=tokenizer,
        template=experiment.template,
        prediction_token=experiment.prediction_token,
        source_pad_token=experiment.source_pad_token or experiment.prediction_token,
        padding_token=experiment.padding_token,
        num_prediction_tokens=experiment.num_prediction_tokens,
        max_length=experiment.max_length,
    )


def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in param_optimizer
                if not any(token in name for token in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                parameter
                for name, parameter in param_optimizer
                if any(token in name for token in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(grouped_parameters, lr=lr)


def train_epoch_bidirectional(
    model: torch.nn.Module,
    train_loader_s2t,
    train_loader_t2s,
    optimizer: torch.optim.Optimizer,
    loss_criterion: nn.Module,
    scheduler,
    device: torch.device,
) -> float:
    model.train(True)
    losses: list[float] = []
    iterator_t2s = iter(train_loader_t2s)

    for data_input_s2t, label_input_s2t in train_loader_s2t:
        for key, value in data_input_s2t.items():
            data_input_s2t[key] = value.to(device)
        for key, value in label_input_s2t.items():
            label_input_s2t[key] = value.to(device)

        try:
            data_input_t2s, label_input_t2s = next(iterator_t2s)
        except StopIteration:
            iterator_t2s = iter(train_loader_t2s)
            data_input_t2s, label_input_t2s = next(iterator_t2s)

        for key, value in data_input_t2s.items():
            data_input_t2s[key] = value.to(device)
        for key, value in label_input_t2s.items():
            label_input_t2s[key] = value.to(device)

        optimizer.zero_grad()
        logits_s2t = model(**data_input_s2t)
        logits_t2s = model(**data_input_t2s)

        target_s2t = label_input_s2t["target"].view(-1)
        target_t2s = label_input_t2s["target"].view(-1)

        logits = torch.cat((logits_s2t, logits_t2s), dim=0)
        targets = torch.cat((target_s2t, target_t2s), dim=0)

        loss = loss_criterion(logits, targets)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
        scheduler.step()

    return sum(losses) / len(losses)


def predict_ids(model: torch.nn.Module, data_loader, device: torch.device) -> list[int]:
    model.eval()
    all_predictions: list[int] = []
    with torch.no_grad():
        for data_input in data_loader:
            for key, value in data_input.items():
                data_input[key] = value.to(device)
            logits = model(**data_input)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs, dim=1)
            all_predictions.extend(predicted.int().cpu().tolist())
    return all_predictions


def train_prompt_model(
    experiment: ExperimentConfig,
    pair: Pair,
    *,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> PromptRunResult:
    seed_everything(experiment.seed)
    run_device = resolve_device(device)
    formatter = build_formatter(experiment)

    train_pairs = read_dictionary(
        str(experiment.train_path(pair)),
        formatter,
        lowercase=experiment.lowercase,
        is_train=True,
        ignore_identical_train_pairs=experiment.ignore_identical_train_pairs,
        max_tokenized_word_length=experiment.max_tokenized_word_length,
    )
    test_pairs = read_dictionary(
        str(experiment.test_path(pair)),
        formatter,
        lowercase=experiment.lowercase,
        is_train=False,
        ignore_identical_train_pairs=False,
        max_tokenized_word_length=experiment.max_tokenized_word_length,
    )

    if train_pairs.empty:
        raise ValueError(f"No train pairs remain after filtering for {pair}.")
    if test_pairs.empty:
        raise ValueError(f"No test pairs remain after filtering for {pair}.")

    train_loader_s2t, eval_loader_s2t, train_loader_t2s, grouped_test = build_dataloaders(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        formatter=formatter,
        batch_size=experiment.batch_size,
        num_workers=experiment.num_workers,
    )

    model = load_model(experiment.pretrained_path, formatter.tokenizer).to(run_device)
    optimizer = build_optimizer(model, experiment.lr)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=formatter.padding_token_id).to(run_device)

    num_train_steps = max(1, len(train_loader_s2t) * experiment.epochs)
    warmup_steps = max(1, int(num_train_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_steps,
    )

    history: list[dict[str, float]] = []
    latest_predictions: list[int] = []
    for epoch in range(experiment.epochs):
        train_loss = train_epoch_bidirectional(
            model,
            train_loader_s2t,
            train_loader_t2s,
            optimizer,
            loss_criterion,
            scheduler,
            run_device,
        )
        latest_predictions = predict_ids(model, eval_loader_s2t, run_device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss})

    saved_checkpoint: Path | None = None
    if checkpoint_path is not None:
        saved_checkpoint = Path(checkpoint_path)
        saved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), saved_checkpoint)

    prompt_predictions = build_prompt_prediction_frame(grouped_test, latest_predictions, formatter)
    metrics = {
        "prompt_p_at_1": p_at_1(prompt_predictions["is_true"]),
        "train_size": float(len(train_pairs)),
        "test_size": float(len(prompt_predictions)),
        "epochs": float(experiment.epochs),
    }
    if history:
        metrics["final_train_loss"] = history[-1]["train_loss"]

    return PromptRunResult(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        prompt_predictions=prompt_predictions,
        metrics=metrics,
        checkpoint_path=saved_checkpoint,
    )
