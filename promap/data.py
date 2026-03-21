from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .templates import PromptTemplate
from .utils import normalize_targets, prediction_head

Direction = Literal["s2t", "t2s"]


@dataclass
class PromptFormatter:
    tokenizer: PreTrainedTokenizerBase
    template: PromptTemplate
    prediction_token: str
    source_pad_token: str
    padding_token: str
    num_prediction_tokens: int
    max_length: int

    def __post_init__(self) -> None:
        self.prediction_token_id = self._single_token_id(self.prediction_token)
        self.source_pad_token_id = self._single_token_id(self.source_pad_token)
        self.padding_token_id = self._single_token_id(self.padding_token)
        self.target_stub = " ".join([self.prediction_token] * self.num_prediction_tokens)

    def _single_token_id(self, token: str) -> int:
        token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Token '{token}' must map to exactly one token id, got {token_ids}.")
        return token_ids[0]

    def word_length(self, word: str) -> int:
        return len(self.tokenizer(word, add_special_tokens=False)["input_ids"])

    def build_prompt(self, source: str, direction: Direction) -> str:
        pad_count = max(0, self.num_prediction_tokens - self.word_length(source))
        source_padding = " ".join([self.source_pad_token] * pad_count)
        if direction == "s2t":
            parts = [self.template.prefix, source, source_padding, self.template.suffix, self.target_stub]
        else:
            parts = [self.template.prefix, self.target_stub, self.template.suffix, source, source_padding]
        return " ".join(part for part in parts if part)

    def encode_prompt(self, source: str, direction: Direction) -> dict[str, torch.Tensor]:
        prompt = self.build_prompt(source, direction)
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].flatten()
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).flatten()
        target_positions = self.find_target_positions(input_ids, direction)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_positions": target_positions,
        }

    def find_target_positions(self, input_ids: torch.Tensor, direction: Direction) -> torch.Tensor:
        positions = torch.nonzero(input_ids == self.prediction_token_id, as_tuple=False).flatten()
        if positions.numel() < self.num_prediction_tokens:
            raise ValueError("Prompt does not contain enough prediction positions.")
        if direction == "s2t":
            return positions[-self.num_prediction_tokens :].to(torch.long)
        return positions[: self.num_prediction_tokens].to(torch.long)

    def encode_target(self, target: str) -> torch.Tensor:
        token_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        padded = token_ids[: self.num_prediction_tokens]
        if len(padded) < self.num_prediction_tokens:
            padded = padded + [self.padding_token_id] * (self.num_prediction_tokens - len(padded))
        return torch.tensor(padded, dtype=torch.long)

    def decode_prediction(self, token_ids: list[int]) -> str:
        filtered = [token_id for token_id in token_ids if int(token_id) != self.padding_token_id]
        if not filtered:
            return ""
        text = self.tokenizer.decode(filtered)
        return text.replace(self.padding_token, "").strip()

    def decode_subtokens(self, token_ids: list[int]) -> list[str]:
        values: list[str] = []
        for token_id in token_ids:
            if int(token_id) == self.padding_token_id:
                continue
            values.append(self.tokenizer.decode([int(token_id)]).strip())
        return values

    def count_padding_predictions(self, token_ids: list[int]) -> int:
        return sum(1 for token_id in token_ids if int(token_id) == self.padding_token_id)


class PromptTrainDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, formatter: PromptFormatter, direction: Direction = "s2t"):
        self.frame = frame.reset_index(drop=True)
        self.formatter = formatter
        self.direction = direction

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        row = self.frame.iloc[index]
        source = row["source"]
        target = row["target"]
        if self.direction == "t2s":
            source, target = target, source
        return self.formatter.encode_prompt(source, self.direction), {
            "target": self.formatter.encode_target(target)
        }


class PromptEvalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, formatter: PromptFormatter, direction: Direction = "s2t"):
        self.frame = frame.reset_index(drop=True)
        self.formatter = formatter
        self.direction = direction
        self.input_column = "source" if direction == "s2t" else "target"

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = self.frame.iloc[index][self.input_column]
        return self.formatter.encode_prompt(source, self.direction)


def read_dictionary(
    dict_path: str,
    formatter: PromptFormatter,
    *,
    lowercase: bool = True,
    is_train: bool = True,
    ignore_identical_train_pairs: bool = True,
    max_tokenized_word_length: int | None = None,
) -> pd.DataFrame:
    pairs: list[tuple[str, str]] = []
    for line in open(dict_path, "r", encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1]
        if lowercase:
            source, target = source.lower(), target.lower()
        if is_train and ignore_identical_train_pairs and source == target:
            continue
        if max_tokenized_word_length is not None:
            if formatter.word_length(source) > max_tokenized_word_length:
                continue
            if formatter.word_length(target) > max_tokenized_word_length:
                continue
        pairs.append((source, target))
    return pd.DataFrame(pairs, columns=["source", "target"])


def group_targets(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("source")["target"].apply(np.array).reset_index(name="target")


def build_dataloaders(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    formatter: PromptFormatter,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    grouped_test = group_targets(test_pairs)
    train_s2t = PromptTrainDataset(train_pairs, formatter, direction="s2t")
    train_t2s = PromptTrainDataset(train_pairs, formatter, direction="t2s")
    eval_s2t = PromptEvalDataset(grouped_test, formatter, direction="s2t")

    return (
        DataLoader(train_s2t, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(eval_s2t, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(train_t2s, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        grouped_test,
    )


def prompt_prediction_is_correct(prediction: str, gold_targets: object) -> int:
    return int(prediction_head(prediction) in normalize_targets(gold_targets))


def build_prompt_prediction_frame(
    grouped_test: pd.DataFrame,
    predicted_ids: list[int],
    formatter: PromptFormatter,
) -> pd.DataFrame:
    chunk_size = formatter.num_prediction_tokens
    chunks = [
        predicted_ids[index : index + chunk_size]
        for index in range(0, len(predicted_ids), chunk_size)
    ]
    if len(chunks) != len(grouped_test):
        raise ValueError("Predictions do not align with grouped test examples.")

    output = grouped_test.copy()
    output["s2t_pred_ids"] = chunks
    output["count_pads"] = output["s2t_pred_ids"].apply(formatter.count_padding_predictions)
    output["s2t_pred_subtokens"] = output["s2t_pred_ids"].apply(formatter.decode_subtokens)
    output["s2t_pred"] = output["s2t_pred_ids"].apply(formatter.decode_prediction)
    output["is_true"] = output.apply(
        lambda row: prompt_prediction_is_correct(row["s2t_pred"], row["target"]),
        axis=1,
    )
    return output
