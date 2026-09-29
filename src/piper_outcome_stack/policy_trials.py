"""Operator task-outcome labels for policy trials and per-position success summaries.

Labels are sidecars (`<trial>.outcome.json`); trial JSON evidence is never rewritten.
"""

import argparse
import json
import math
from pathlib import Path

OUTCOMES = ("success", "failure", "invalid")


def label_path(trial):
    trial = Path(trial)
    return trial.with_name(trial.stem + ".outcome.json")


def label(trial, outcome, position=None, note=None):
    trial = Path(trial)
    report = json.loads(trial.read_text())
    position = position or report.get("position")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    if not position:
        raise ValueError("trial has no position label; pass --position")
    value = dict(
        trial=trial.name,
        position=position,
        task_outcome=outcome,
        note=note,
        program_status=report.get("status"),
        end_reason=report.get("end_reason"),
        policy_actions_sent=report.get("policy_actions_sent"),
    )
    with label_path(trial).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return value


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(paths):
    """Invalid attempts are reported separately and excluded from the rate denominator."""
    cells = {}
    for path in paths:
        value = json.loads(Path(path).read_text())
        cell = cells.setdefault(value["position"], dict(success=0, failure=0, invalid=0))
        cell[value["task_outcome"]] += 1
    rows = {}
    for position, cell in sorted(cells.items()):
        n = cell["success"] + cell["failure"]
        rows[position] = dict(
            **cell,
            attempts=n,
            success_rate=None if n == 0 else cell["success"] / n,
            wilson95=wilson(cell["success"], n),
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("label")
    add.add_argument("trial", type=Path)
    add.add_argument("outcome", choices=OUTCOMES)
    add.add_argument("--position")
    add.add_argument("--note")
    report = commands.add_parser("summarize")
    report.add_argument("labels", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "label":
        result = label(args.trial, args.outcome, args.position, args.note)
    else:
        result = summarize(args.labels)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
