from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .templates import PromptTemplate, resolve_template
from .utils import maybe_resolve_path

Pair = tuple[str, str]


@dataclass
class ExperimentConfig:
    name: str
    pretrained_path: str
    pairs: list[Pair]
    data_root: Path
    train_path_template: str = "{src}_{tgt}_train.txt"
    test_path_template: str = "{src}_{tgt}_test.txt"
    template_language: str = "en"
    template_prefix: str | None = None
    template_suffix: str | None = None
    prediction_token: str = "<special1>"
    source_pad_token: str | None = None
    padding_token: str = "<pad>"
    num_prediction_tokens: int = 4
    max_length: int = 20
    epochs: int = 5
    lr: float = 2e-5
    batch_size: int = 64
    num_workers: int = 1
    seed: int = 42
    lowercase: bool = True
    ignore_identical_train_pairs: bool = True
    max_tokenized_word_length: int | None = None
    output_dir: Path = Path("outputs")
    checkpoint_dir: str = "checkpoints"
    prompt_predictions_dir: str = "prompt_predictions"
    reranked_predictions_dir: str = "reranked_predictions"
    similarity_candidates_dir: Path | None = None
    similarity_path_template: str | None = None
    similarity_temperature: float = 0.1
    similarity_top_k: int = 10
    store_intermediate: bool = True

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        if self.source_pad_token is None:
            self.source_pad_token = self.prediction_token
        if self.max_tokenized_word_length is None:
            self.max_tokenized_word_length = self.num_prediction_tokens

    @property
    def template(self) -> PromptTemplate:
        return resolve_template(
            language=self.template_language,
            prefix=self.template_prefix,
            suffix=self.template_suffix,
        )

    def pair_name(self, pair: Pair) -> str:
        return f"{pair[0]}_{pair[1]}"

    def train_path(self, pair: Pair) -> Path:
        return self.data_root / self.train_path_template.format(src=pair[0], tgt=pair[1])

    def test_path(self, pair: Pair) -> Path:
        return self.data_root / self.test_path_template.format(src=pair[0], tgt=pair[1])

    def similarity_candidates_path(self, pair: Pair) -> Path | None:
        if not self.similarity_candidates_dir or not self.similarity_path_template:
            return None
        return self.similarity_candidates_dir / self.similarity_path_template.format(
            src=pair[0],
            tgt=pair[1],
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    base_dir = config_path.parent

    pairs = [tuple(item) for item in raw["pairs"]]
    return ExperimentConfig(
        name=raw["name"],
        pretrained_path=raw["pretrained_path"],
        pairs=pairs,
        data_root=maybe_resolve_path(base_dir, raw["data_root"]),
        train_path_template=raw.get("train_path_template", "{src}_{tgt}_train.txt"),
        test_path_template=raw.get("test_path_template", "{src}_{tgt}_test.txt"),
        template_language=raw.get("template_language", "en"),
        template_prefix=raw.get("template_prefix"),
        template_suffix=raw.get("template_suffix"),
        prediction_token=raw.get("prediction_token", "<special1>"),
        source_pad_token=raw.get("source_pad_token"),
        padding_token=raw.get("padding_token", "<pad>"),
        num_prediction_tokens=raw.get("num_prediction_tokens", 4),
        max_length=raw.get("max_length", 20),
        epochs=raw.get("epochs", 5),
        lr=raw.get("lr", 2e-5),
        batch_size=raw.get("batch_size", 64),
        num_workers=raw.get("num_workers", 1),
        seed=raw.get("seed", 42),
        lowercase=raw.get("lowercase", True),
        ignore_identical_train_pairs=raw.get("ignore_identical_train_pairs", True),
        max_tokenized_word_length=raw.get("max_tokenized_word_length"),
        output_dir=maybe_resolve_path(base_dir, raw.get("output_dir", "outputs")),
        checkpoint_dir=raw.get("checkpoint_dir", "checkpoints"),
        prompt_predictions_dir=raw.get("prompt_predictions_dir", "prompt_predictions"),
        reranked_predictions_dir=raw.get("reranked_predictions_dir", "reranked_predictions"),
        similarity_candidates_dir=maybe_resolve_path(base_dir, raw.get("similarity_candidates_dir")),
        similarity_path_template=raw.get("similarity_path_template"),
        similarity_temperature=raw.get("similarity_temperature", 0.1),
        similarity_top_k=raw.get("similarity_top_k", 10),
        store_intermediate=raw.get("store_intermediate", True),
    )
