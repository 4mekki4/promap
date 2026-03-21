from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from promap.workflows import summarize_local_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the saved Promap pickles in this workspace.")
    parser.add_argument("--preds-dir", default="../preds")
    parser.add_argument("--arabic-preds-dir", default="../arabic_preds")
    parser.add_argument("--output", default=None, help="Optional output directory for CSV files")
    args = parser.parse_args()

    promap_summary, arabic_summary = summarize_local_results(
        args.preds_dir,
        args.arabic_preds_dir,
        output_dir=args.output,
    )

    print("Promap results")
    if promap_summary.empty:
        print("  no pickles found")
    else:
        print(promap_summary.to_string(index=False))

    print("\nArabic results")
    if arabic_summary.empty:
        print("  no pickles found")
    else:
        print(arabic_summary.to_string(index=False))


if __name__ == "__main__":
    main()
