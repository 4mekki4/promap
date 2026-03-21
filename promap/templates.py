from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    prefix: str
    suffix: str


DEFAULT_TEMPLATES: dict[str, PromptTemplate] = {
    "en": PromptTemplate(prefix="the translation of the word", suffix="is"),
    "fr": PromptTemplate(prefix="La traduction du mot", suffix="est"),
    "ar": PromptTemplate(prefix="ترجمة كلمة", suffix="هي"),
}


def resolve_template(
    language: str = "en",
    prefix: str | None = None,
    suffix: str | None = None,
) -> PromptTemplate:
    if prefix is not None or suffix is not None:
        if not prefix or not suffix:
            raise ValueError("Custom templates require both a prefix and a suffix.")
        return PromptTemplate(prefix=prefix, suffix=suffix)
    try:
        return DEFAULT_TEMPLATES[language]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_TEMPLATES))
        raise ValueError(f"Unknown template language '{language}'. Known templates: {known}.") from exc
