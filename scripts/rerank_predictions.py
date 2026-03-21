from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from promap.workflows import rerank_saved_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank saved prompt predictions with similarity candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-predictions", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metrics = rerank_saved_predictions(
        args.config,
        checkpoint=args.checkpoint,
        prompt_predictions=args.prompt_predictions,
        candidates=args.candidates,
        output=args.output,
        metrics_output=args.metrics_output,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
