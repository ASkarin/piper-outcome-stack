"""Record one operator result after a development trial has stopped."""

from pathlib import Path
import json
import sys


def prompt(trial, input_fn=input):
    trial = Path(trial)
    output = trial.with_name(trial.stem + ".outcome.json")
    if output.exists():
        return json.loads(output.read_text())
    report = json.loads(trial.read_text())
    answer = input_fn("\r结果：1成功 / 2失败 / 3外部无效重测：").strip()
    if answer not in ("1", "2", "3"):
        raise ValueError("未保存标签；请重新执行 label-trial.py 补录。")
    clean = (
        report.get("status") == "segment_ended_held"
        and report.get("stop_outcome") == "hold_confirmed"
    )
    if answer in ("1", "3") and not clean:
        raise ValueError("程序未正常结束并确认保持，不能标成功或外部无效。")
    note = input_fn("\r外部操作／摆放原因：").strip() if answer == "3" else None
    if answer == "3" and not note:
        raise ValueError("外部无效必须说明原因；原记录保留。")
    value = {
        "trial": trial.name,
        "position": report["position"],
        "task_outcome": {"1": "success", "2": "failure", "3": "invalid"}[answer],
        "note": note,
        "program_status": report.get("status"),
        "stop_outcome": report.get("stop_outcome"),
        "end_reason": report.get("end_reason"),
        "policy_actions_sent": report.get("policy_actions_sent"),
        "scope": "development",
    }
    with output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print("\r结果已保存。" if answer != "3" else "\r无效尝试已保留；重新运行同一命令可重测。")
    return value


if __name__ == "__main__":
    prompt(sys.argv[1])
