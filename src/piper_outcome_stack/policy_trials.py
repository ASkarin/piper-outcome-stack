"""Operator task-outcome labels for policy trials and per-position success summaries.

Labels are sidecars (`<trial>.outcome.json`); trial JSON evidence is never rewritten.
"""

import argparse
import json
import math
import sys
from pathlib import Path

OUTCOMES = ("success", "failure", "invalid")


def label_path(trial):
    trial = Path(trial)
    return trial.with_name(trial.stem + ".outcome.json")


def label(trial, outcome, position=None, note=None, interventions=None):
    trial = Path(trial)
    report = json.loads(trial.read_text())
    position = position or report.get("position")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    if not position:
        raise ValueError("trial has no position label; pass --position")
    context = report.get("task_context")
    if context and context.get("task_family") == "C":
        if report.get("mode") != "run":
            raise ValueError("C software checks are not task trials")
        if outcome == "success" and (
            report.get("status") != "segment_ended_held" or not report.get("hold_confirmed")
        ):
            raise ValueError("C success cannot override a program failure or unconfirmed hold")
        if report.get("status") == "failed" and outcome != "failure":
            raise ValueError("C program failures must remain failed trials")
    value = dict(
        trial=trial.name,
        position=position,
        task_outcome=outcome,
        note=note,
        program_status=report.get("status"),
        end_reason=report.get("end_reason"),
        policy_actions_sent=report.get("policy_actions_sent"),
    )
    if context:
        value.update(task_context=context, task=report["task"], interventions=interventions or [])
    with label_path(trial).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return value


def prompt_label(trial, input_fn=input, output_fn=print):
    """Ask only after a trial exits; never starts or accesses robot devices."""
    trial = Path(trial)
    report = json.loads(trial.read_text())
    if label_path(trial).exists():
        value = json.loads(label_path(trial).read_text())
        output_fn("本轮已标注为 " + value["task_outcome"] + "，保留原标注。")
        return value
    output_fn("试跑程序已结束。位置：" + str(report.get("position", "未记录")))
    output_fn(
        "程序状态：" + str(report.get("status")) + "；停止结果：" + str(report.get("stop_outcome"))
    )
    context = report.get("task_context")
    if context and context.get("task_family") == "C":
        output_fn("程序状态与物理观察分别保留；程序故障回合计失败。")
        output_fn("本次只评价单技能，不代表完整C任务。")
        output_fn("1 成功：" + context["success_description"])
    else:
        output_fn("请按实际任务结果标注，程序报错不自动等于抓放失败。")
        output_fn("1 成功：抓起、搬运、放下并松爪全部完成。")
    output_fn("2 失败：任务未全部完成。")
    output_fn("3 废次：摆放等外部条件不符合要求，必须说明原因。")
    output_fn("回车跳过：保留日志，稍后标注；不会自动开始下一轮。")
    choices = {
        "1": "success",
        "2": "failure",
        "3": "invalid",
        "success": "success",
        "failure": "failure",
        "invalid": "invalid",
    }
    try:
        while True:
            answer = input_fn("结果 [1/2/3/回车跳过]：").strip().lower()
            if not answer:
                output_fn("本轮暂未标注，原始日志已保留。")
                return None
            if answer in choices:
                break
            output_fn("请输入1、2、3，或直接回车跳过。")
        outcome = choices[answer]
        note = ""
        interventions = []
        if context and context.get("task_family") == "C":
            intervention = input_fn("中途人工调整/接管（没有则回车）：").strip()
            if intervention:
                interventions.append(intervention)
        if outcome != "success":
            note = input_fn("失败阶段或废次原因：").strip()
            while outcome == "invalid" and not note:
                note = input_fn("废次必须注明外部原因；未完成抓放应选失败。原因：").strip()
    except (EOFError, KeyboardInterrupt):
        output_fn("本轮暂未标注，原始日志已保留。")
        return None
    return label(trial, outcome, note=note or None, interventions=interventions)


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
        context = value.get("task_context")
        key = value["position"]
        if context and context.get("task_family") == "C":
            key = "/".join(
                [context["scene_id"], context["skill_id"], context.get("object_id", "-")]
            )
        cell = cells.setdefault(key, dict(success=0, failure=0, invalid=0))
        cell[value["task_outcome"]] += 1
        if context and context.get("task_family") == "C":
            cell["assisted"] = cell.get("assisted", 0) + bool(value.get("interventions"))
            cell["unassisted_successes"] = cell.get("unassisted_successes", 0) + (
                value["task_outcome"] == "success" and not value.get("interventions")
            )
    rows = {}
    for position, cell in sorted(cells.items()):
        n = cell["success"] + cell["failure"]
        rows[position] = dict(
            **cell,
            attempts=n,
            success_rate=None if n == 0 else cell["success"] / n,
            wilson95=wilson(cell["success"], n),
        )
        if "unassisted_successes" in cell:
            rows[position].update(
                scope="single_skill_operator_stopped",
                unassisted_success_rate=None if n == 0 else cell["unassisted_successes"] / n,
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
    add.add_argument("--intervention", action="append")
    prompt = commands.add_parser("prompt")
    prompt.add_argument("trial", type=Path)
    report = commands.add_parser("summarize")
    report.add_argument("labels", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "label":
        result = label(args.trial, args.outcome, args.position, args.note, args.intervention)
    elif args.command == "prompt":
        if sys.stdout.isatty():
            result = prompt_label(
                args.trial,
                input_fn=lambda message: input("\r" + message),
                output_fn=lambda message: print(message, end="\r\n", flush=True),
            )
        else:
            result = prompt_label(args.trial)
    else:
        result = summarize(args.labels)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.command == "prompt" and sys.stdout.isatty():
        print(rendered.replace("\n", "\r\n"), end="\r\n", flush=True)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
