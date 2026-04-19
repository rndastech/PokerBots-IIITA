#!/usr/bin/env python3
"""Qualification gatekeeper for PokerBots submissions.

This script validates all bots under submission/ and runs each one against
baseline_bot over repeated matches. A bot is accepted only if it reaches the
minimum win rate threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tournament_utils import discover_submission_bots, run_isolated_match, validate_submission


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown_summary(results: list[dict], args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("<!-- pokerbots-qualification-report -->")
    lines.append("## PokerBots Qualification Report")
    lines.append("")
    lines.append(f"Submissions root: `{args.submissions_root}`")
    lines.append(f"Baseline bot: `{args.baseline_path}`")
    lines.append(f"Hands per match: **{args.hands_per_match}**")
    lines.append(f"Qualification rounds per bot: **{args.qualification_rounds}**")
    lines.append(f"Minimum win rate: **{args.min_win_rate:.0%}**")

    if not results:
        lines.append("")
        lines.append("No bots were discovered under `submission/<roll_no>/(python_bot|cpp_bot)`.")
        return "\n".join(lines)

    lines.append("")
    lines.append("### Per-Submission Results")
    lines.append("| Submission | Validation | Wins | Losses | Draws | Win Rate | Qualified |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for entry in results:
        validation = "PASS" if entry["validation_ok"] else "FAIL"
        verdict = "PASS" if entry["qualified"] else "FAIL"
        lines.append(
            "| {submission} | {validation} | {wins} | {losses} | {draws} | {win_rate:.2%} | {verdict} |".format(
                submission=entry["bot_id"],
                validation=validation,
                wins=entry["wins"],
                losses=entry["losses"],
                draws=entry["draws"],
                win_rate=entry["win_rate"],
                verdict=verdict,
            )
        )

    failing = [row for row in results if not row["qualified"]]
    if failing:
        lines.append("")
        lines.append("### Failures")
        for row in failing:
            lines.append(f"- **{row['bot_id']}**: {row['notes']}")

    return "\n".join(lines)


def _write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    result_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_md = _render_markdown_summary(result_rows, args)
    (output_dir / "summary.md").write_text(summary_md + "\n", encoding="utf-8")

    _write_csv(
        output_dir / "results.csv",
        result_rows,
        [
            "bot_id",
            "submission_path",
            "validation_ok",
            "matches_attempted",
            "matches_ok",
            "wins",
            "losses",
            "draws",
            "win_rate",
            "qualified",
            "notes",
        ],
    )

    payload = {
        "submissions_root": args.submissions_root,
        "baseline_path": args.baseline_path,
        "hands_per_match": args.hands_per_match,
        "qualification_rounds": args.qualification_rounds,
        "min_win_rate": args.min_win_rate,
        "results": result_rows,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run qualification against baseline bot")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--submissions-root", default="submission", help="Submission root directory")
    parser.add_argument("--baseline-path", default="baseline_bot", help="Baseline bot directory")
    parser.add_argument(
        "--hands-per-match",
        type=int,
        default=500,
        help="Hands per match against baseline bot",
    )
    parser.add_argument(
        "--qualification-rounds",
        type=int,
        default=100,
        help="Number of matches each submission plays vs baseline",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.60,
        help="Minimum win rate required for acceptance",
    )
    parser.add_argument(
        "--output-dir",
        default=".qualification",
        help="Directory where markdown/json/csv summaries and logs are written",
    )
    args = parser.parse_args()

    if args.hands_per_match <= 0:
        raise SystemExit("--hands-per-match must be > 0")
    if args.qualification_rounds <= 0:
        raise SystemExit("--qualification-rounds must be > 0")
    if not 0.0 <= args.min_win_rate <= 1.0:
        raise SystemExit("--min-win-rate must be in [0.0, 1.0]")

    repo_root = Path(args.repo_root).resolve()
    submissions_root = (repo_root / args.submissions_root).resolve()
    baseline_abs = (repo_root / args.baseline_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []

    if not baseline_abs.is_dir():
        row = {
            "bot_id": "BASELINE_MISSING",
            "submission_path": args.baseline_path,
            "validation_ok": False,
            "matches_attempted": 0,
            "matches_ok": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
            "qualified": False,
            "notes": f"Baseline directory not found: {args.baseline_path}",
        }
        result_rows.append(row)
        _write_outputs(output_dir, args=args, result_rows=result_rows)
        return 2

    if not submissions_root.is_dir():
        row = {
            "bot_id": "SUBMISSION_ROOT_MISSING",
            "submission_path": args.submissions_root,
            "validation_ok": False,
            "matches_attempted": 0,
            "matches_ok": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
            "qualified": False,
            "notes": f"Submissions directory not found: {args.submissions_root}",
        }
        result_rows.append(row)
        _write_outputs(output_dir, args=args, result_rows=result_rows)
        return 2

    submissions = discover_submission_bots(submissions_root)
    if not submissions:
        _write_outputs(output_dir, args=args, result_rows=result_rows)
        return 1

    for submission in submissions:
        row = {
            "bot_id": submission.bot_id,
            "submission_path": submission.path.as_posix(),
            "validation_ok": False,
            "matches_attempted": args.qualification_rounds,
            "matches_ok": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
            "qualified": False,
            "notes": "",
        }

        validation = validate_submission(submission, repo_root)
        row["validation_ok"] = validation.ok
        if not validation.ok:
            row["notes"] = " | ".join(validation.errors)
            result_rows.append(row)
            continue

        submission_abs = (repo_root / submission.path).resolve()
        bot_log_dir = logs_dir / _sanitize_name(submission.bot_id)
        failure_reasons: list[str] = []

        for round_index in range(args.qualification_rounds):
            match_name = f"{_sanitize_name(submission.bot_id)}_r{round_index + 1}"
            match = run_isolated_match(
                repo_root=repo_root,
                player_1_source=baseline_abs,
                player_2_source=submission_abs,
                output_dir=bot_log_dir,
                player_1_name="BASELINE",
                player_2_name=match_name,
                num_rounds=args.hands_per_match,
                timeout_seconds=1200,
            )

            if not match.ok:
                row["losses"] += 1
                reason = match.failure_reason or "Unknown match failure"
                if len(failure_reasons) < 5:
                    failure_reasons.append(reason)
                continue

            row["matches_ok"] += 1
            if match.player_2_bankroll > match.player_1_bankroll:
                row["wins"] += 1
            elif match.player_2_bankroll < match.player_1_bankroll:
                row["losses"] += 1
            else:
                row["draws"] += 1

        row["win_rate"] = row["wins"] / args.qualification_rounds
        row["qualified"] = row["win_rate"] >= args.min_win_rate

        if failure_reasons:
            row["notes"] = " | ".join(failure_reasons)
        elif not row["qualified"]:
            row["notes"] = (
                f"Win rate below threshold: {row['win_rate']:.2%} < {args.min_win_rate:.2%}"
            )

        result_rows.append(row)

    _write_outputs(output_dir, args=args, result_rows=result_rows)

    qualified_all = bool(result_rows) and all(row.get("qualified", False) for row in result_rows)
    return 0 if qualified_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
