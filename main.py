#!/usr/bin/env python3
"""Command line front end for the lyrics attribution engine.

Examples
--------
    python3 main.py --query data/query_song.txt
    python3 main.py --query data/query_song.txt --k 4 --top 5
    python3 main.py --query data/query_song.txt --json report.json
    python3 main.py --text "so carry me home through the blue and the rain"
"""

import argparse
import json
import os
import sys

from engine import AttributionEngine

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATALOGUE = os.path.join(HERE, "data", "catalogue.json")


def build_parser():
    p = argparse.ArgumentParser(description="Lyrics similarity & attribution engine")
    p.add_argument("--catalogue", default=DEFAULT_CATALOGUE,
                   help="JSON catalogue file (default: data/catalogue.json)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--query", help="path to a query lyrics file")
    src.add_argument("--text", help="query lyrics given inline")
    p.add_argument("--k", type=int, default=5, help="shingle size in tokens (default 5)")
    p.add_argument("--top", type=int, default=3, help="candidates to report (default 3)")
    p.add_argument("--min-run", type=int, default=2,
                   help="minimum consecutive shingles to flag a passage (default 2)")
    p.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    return p


def print_report(report):
    stats = report.query_stats
    print()
    print("=" * 68)
    print(f"  ATTRIBUTION REPORT  ::  {report.query_title}")
    print("=" * 68)
    print(f"  query tokens: {stats['tokens']}   "
          f"shingles (k={stats['k']}): {stats['shingles']}   "
          f"distinct: {stats['distinct_shingles']}")

    if not report.matches:
        print("\n  No candidate songs shared a single passage with the query.\n")
        return

    for rank, match in enumerate(report.matches, start=1):
        s = match.signals
        print("\n" + "-" * 68)
        print(f"  #{rank}  {match.song.label()}   [{match.song.id}]")
        print(f"      score    {match.score:.3f}   ->  {match.verdict.upper()}")
        print(f"      signals  containment {s['containment']:.3f} | "
              f"jaccard {s['jaccard']:.3f} | lcs {s['lcs_ratio']:.3f}")

        if match.passages:
            print(f"      flagged passages: {len(match.passages)}")
            for p in match.passages:
                ql = _fmt_lines(p.query_lines)
                sl = _fmt_lines(p.source_lines)
                print(f"        • query {ql} ~ source {sl}  ({p.tokens} tokens)")
                print(f"            query : {p.query_text}")
                print(f"            source: {p.source_text}")
        else:
            print("      flagged passages: none above threshold")

    best = report.best
    print("\n" + "=" * 68)
    print(f"  MOST LIKELY SOURCE: {best.song.label()} "
          f"({best.score:.1%} similarity, {best.verdict})")
    print("=" * 68 + "\n")


def _fmt_lines(span):
    a, b = span
    return f"L{a}" if a == b else f"L{a}-{b}"


def main(argv=None):
    args = build_parser().parse_args(argv)

    engine = AttributionEngine(k=args.k, top_k=args.top,
                               min_run_shingles=args.min_run)
    stats = engine.ingest_json(args.catalogue)
    print(f"Indexed {stats['songs']} songs | {stats['total_tokens']} tokens | "
          f"{stats['distinct_fingerprints']} distinct fingerprints")

    if args.query:
        report = engine.attribute_file(args.query)
    else:
        report = engine.attribute(args.text, title="inline query")

    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"JSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
