"""Operator RTC trials for two frozen50k models; check mode opens no devices."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["check", "run"])
parser.add_argument("model", choices=["original", "flat"])
parser.add_argument("skill", choices=["green-left", "green-right", "blue-left", "blue-right"])
parser.add_argument("layout", choices=["1", "2"])
args = parser.parse_args()
operator = json.loads((base / "operator-config.json").read_text())
policy = operator["policies"][args.skill]
directory = base / ("checks" if args.mode == "check" else "trials")
directory.mkdir(exist_ok=True)
run = Path(
    tempfile.mkdtemp(prefix=f"{args.model}-{args.skill}-layout{args.layout}-", dir=directory)
)
output = run / "trial.json"
command = [
    operator["python"],
    str(base / "piper_policy_trial.py"),
    args.mode,
    "--policy-type",
    "smolvla",
    "--task",
    policy["instruction"],
    "--vlm-assets",
    operator["vlm_assets"],
    "--seed",
    "42",
    "--joint-representation",
    "absolute",
    "--position",
    "B-layout" + args.layout,
    "--duration-s",
    "60",
    "--record-rgb",
    "--record-action-chunks",
    "--n-action-steps",
    "25",
    "--config",
    policy["config"],
    "--checkpoint",
    operator["models"][args.model]["checkpoint"],
    "--reference-inputs",
    str(base / "reference-inputs" / (args.model + "-" + args.skill + ".npz")),
    "--output",
    str(output),
]
env = {
    **os.environ,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "NVIDIA_TF32_OVERRIDE": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "PYTHONPATH": str(base / "source"),
    "PYTHONDONTWRITEBYTECODE": "1",
}
if args.mode == "run":
    keys = [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
    ]
    command = [
        "sudo",
        "piper-socketcan",
        "exec",
        "--",
        "env",
        *[f"{k}={env[k]}" for k in keys],
        *command,
    ]
(run / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n")
with (run / "console.log").open("xb") as log:
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
    )
    previous_cr = False
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        log.write(chunk)
        log.flush()
        shown = bytearray()
        for value in chunk:
            if sys.stdout.isatty() and value == 10 and not previous_cr:
                shown.append(13)
            shown.append(value)
            previous_cr = value == 13
        sys.stdout.buffer.write(shown)
        sys.stdout.buffer.flush()
    code = process.wait()
(run / "exit-code.json").write_text(json.dumps({"exit_code": code}) + "\n")
if args.mode == "run" and output.exists() and output.stat().st_size:
    subprocess.run([sys.executable, str(base / "label-trial.py"), str(output)], check=True)
raise SystemExit(code)
