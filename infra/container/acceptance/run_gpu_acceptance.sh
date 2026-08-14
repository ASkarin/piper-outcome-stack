#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

mapfile -t gpu_uuids < <(
    nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)
[[ "${#gpu_uuids[@]}" -eq 3 ]] || {
    echo "expected exactly three GPUs" >&2
    exit 2
}

if [[ "${1:-}" != "--locked" ]]; then
    run_id="environment-acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
    gpu_csv="$(IFS=,; echo "${gpu_uuids[*]}")"
    exec piper-gpu-run \
        --gpus "${gpu_csv}" \
        --run-id "${run_id}" \
        --repo "${REPO_ROOT}" \
        -- \
        bash "${BASH_SOURCE[0]}" --locked
fi
shift

[[ -n "${PIPER_RUN_DIR:-}" ]] || {
    echo "--locked is reserved for piper-gpu-run" >&2
    exit 2
}
readonly OUTPUT_ROOT="${PIPER_ACCEPTANCE_OUTPUT:-${PIPER_RUN_DIR}/acceptance}"

mkdir -p "${OUTPUT_ROOT}"

piper-env-doctor --repo "${REPO_ROOT}" --json \
    | tee "${OUTPUT_ROOT}/environment-doctor.json"
nvidia-smi topo -m | tee "${OUTPUT_ROOT}/gpu-topology.txt"
numactl --hardware | tee "${OUTPUT_ROOT}/numa-topology.txt"
python "${SCRIPT_DIR}/image_smoke.py" | tee "${OUTPUT_ROOT}/image-smoke.json"
python "${SCRIPT_DIR}/logging_smoke.py" | tee "${OUTPUT_ROOT}/logging-smoke.json"

for gpu_uuid in "${gpu_uuids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" python - "${gpu_uuid}" <<'PY' \
        | tee "${OUTPUT_ROOT}/single-${gpu_uuid}.json"
import json
import sys

import torch

tensor = torch.ones(1024, device="cuda")
result = {
    "gpu_uuid": sys.argv[1],
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "sum": tensor.sum().item(),
    "status": "pass",
}
print(json.dumps(result, indent=2, sort_keys=True))
PY
done

CUDA_VISIBLE_DEVICES="${gpu_uuids[0]},${gpu_uuids[1]}" \
    torchrun --standalone --nproc-per-node=2 \
    "${SCRIPT_DIR}/nccl_allreduce.py" --iterations 100 \
    | tee "${OUTPUT_ROOT}/nccl-two-gpu.json"

CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_uuids[*]}")" \
    torchrun --standalone --nproc-per-node=3 \
    "${SCRIPT_DIR}/nccl_allreduce.py" --iterations 100 \
    | tee "${OUTPUT_ROOT}/nccl-three-gpu.json"

CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_uuids[*]}")" \
    torchrun --standalone --nproc-per-node=3 \
    "${SCRIPT_DIR}/dataloader_smoke.py" \
        --duration-seconds 600 \
        --workers 2 \
    | tee "${OUTPUT_ROOT}/dataloader-three-rank.jsonl"

echo "acceptance outputs: ${OUTPUT_ROOT}"
