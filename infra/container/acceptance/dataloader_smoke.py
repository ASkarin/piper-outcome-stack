from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticFrames(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "rgb": torch.rand((3, 224, 224), generator=generator),
            "state": torch.rand((32,), generator=generator),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="PiPER DataLoader shared-memory acceptance check")
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    loader = DataLoader(
        SyntheticFrames(1_000_000),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    started = time.monotonic()
    batches = 0
    samples = 0
    while time.monotonic() - started < args.duration_seconds:
        for batch in loader:
            if torch.cuda.is_available():
                batch["rgb"].to(device, non_blocking=True)
                batch["state"].to(device, non_blocking=True)
                torch.cuda.synchronize()
            batches += 1
            samples += batch["rgb"].shape[0]
            if time.monotonic() - started >= args.duration_seconds:
                break

    payload = {
        "schema_version": 1,
        "rank": int(os.environ.get("RANK", "0")),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "duration_seconds": time.monotonic() - started,
        "batches": batches,
        "samples": samples,
        "status": "pass",
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
