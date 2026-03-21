from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ProMapModel(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(pretrained_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_positions: torch.Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )
        sequence = outputs.hidden_states[-1] if return_hidden_states else outputs.logits
        batch_indices = torch.arange(sequence.size(0), device=sequence.device).unsqueeze(1)
        gathered = sequence[batch_indices, target_positions]
        return gathered.reshape(-1, sequence.size(-1))


def load_tokenizer(pretrained_path: str, extra_tokens: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    unique_tokens = list(dict.fromkeys(token for token in extra_tokens if token))
    if unique_tokens:
        tokenizer.add_tokens(unique_tokens)
    return tokenizer


def load_model(pretrained_path: str, tokenizer) -> ProMapModel:
    model = ProMapModel(pretrained_path)
    embedding_count = model.transformer.get_input_embeddings().num_embeddings
    if len(tokenizer) != embedding_count:
        model.transformer.resize_token_embeddings(len(tokenizer))
    return model


def load_model_checkpoint(pretrained_path: str, tokenizer, checkpoint_path: str | Path) -> ProMapModel:
    model = load_model(pretrained_path, tokenizer)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model
