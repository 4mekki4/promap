from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from promap.workflows import run_scenario


def parse_pair(value: str) -> tuple[str, str]:
    normalized = value.replace("-", "_")
    parts = normalized.split("_")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected a pair like 'en-fr' or 'en_fr', got '{value}'."
        )
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ProMap experiment config as plain Python.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--pairs", nargs="*", type=parse_pair, help="Optional subset of pairs")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--skip-rerank", action="store_true", help="Only run the prompt model")
    parser.add_argument("--candidate-dir", default=None, help="Override the candidate directory")
    parser.add_argument(
        "--candidate-template",
        default=None,
        help="Override the candidate filename template, e.g. '{src}_{tgt}_candidates.tsv'",
    )
    args = parser.parse_args()
    run_scenario(
        args.config,
        pairs=args.pairs,
        device=args.device,
        skip_rerank=args.skip_rerank,
        candidate_dir=args.candidate_dir,
        candidate_template=args.candidate_template,
    )


if __name__ == "__main__":
    main()
