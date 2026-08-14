from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="PiPER NCCL all-reduce acceptance check")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tensor-mib", type=int, default=64)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    element_count = args.tensor_mib * 1024 * 1024 // 4
    tensor = torch.ones(element_count, device=device, dtype=torch.float32)

    for _ in range(args.warmup):
        tensor.fill_(1)
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    durations_ms: list[float] = []
    for _ in range(args.iterations):
        tensor.fill_(1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - started) * 1000)

    expected = float(world_size)
    if not torch.isfinite(tensor).all():
        raise RuntimeError("all-reduce result contains non-finite values")
    if tensor[0].item() != expected:
        raise RuntimeError(f"unexpected all-reduce result {tensor[0].item()} != {expected}")

    if rank == 0:
        tensor_bytes = tensor.numel() * tensor.element_size()
        median_seconds = statistics.median(durations_ms) / 1000
        algorithm_gbps = tensor_bytes / median_seconds / 1e9
        payload = {
            "schema_version": 1,
            "world_size": world_size,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "tensor_mib": args.tensor_mib,
            "median_ms": statistics.median(durations_ms),
            "p95_ms": percentile(durations_ms, 0.95),
            "algorithm_gbps": algorithm_gbps,
            "status": "pass",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
