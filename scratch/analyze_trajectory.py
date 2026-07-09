"""CLI wrapper for trajectory analysis.

Core logic lives in edel.analysis.trajectory — this script handles
argument parsing, report formatting, and artifact persistence.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from edel.analysis.trajectory import analyze_trajectory, format_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse the epistemic trajectory of a paper and compare its neighbours."
    )
    parser.add_argument("--file", required=True, help="Path to the dimensionality reduction parquet file.")
    parser.add_argument("--paper_id", required=True, help="Target paper ID (e.g., https://openalex.org/W123456).")
    parser.add_argument("--method", default="diffusion", help="Projection method (diffusion or umap).")
    parser.add_argument(
        "--space",
        choices=["2d", "embedding"],
        default="embedding",
        help="Distance space: '2d' (Euclidean on projection) or 'embedding' (cosine on raw vectors).",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbours to retrieve.")
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius threshold (overrides --k when set).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/trajectory_report.md",
        help="Output path for the Markdown report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.file}...")
    df = pd.read_parquet(args.file)

    if args.paper_id not in df["id"].values:
        print(f"Error: Paper ID '{args.paper_id}' not found in the dataset.")
        sys.exit(1)

    print(f"Analysing trajectory for: {args.paper_id}")
    for asp in ["problem", "method", "finding", "interpretation"]:
        print(f"  Processing aspect: {asp}...")

    result = analyze_trajectory(
        df,
        paper_id=args.paper_id,
        space=args.space,
        method=args.method,
        k=args.k,
        radius=args.radius,
    )

    report_text = format_report(result, k=args.k, radius=args.radius)

    print("\n" + "=" * 80 + "\n")
    print(report_text)
    print("=" * 80 + "\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)

    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
