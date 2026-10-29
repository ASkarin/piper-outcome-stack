"""Export saved raw predictions and matching recorded inputs; zero new inference."""

from pathlib import Path
import json
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STAGE = Path("/workspace/piper/staging/csm/smolvla-retrain-20261027")
OUTPUT = STAGE / "rtc-reference-20261027"
TASKS = {
    "green-left": "把绿色方块放到左区。",
    "green-right": "把绿色方块放到右区。",
    "blue-left": "把天蓝色六边形块放到左区。",
    "blue-right": "把天蓝色六边形块放到右区。",
}


def main():
    OUTPUT.mkdir(exist_ok=False)
    cache = {}
    rows = []
    data = {}
    samples = None
    for arm in ["original", "flat"]:
        root = Path("/workspace/piper/runs/csm") / f"smolvla-b-{arm}-fresh-20261027-seed42"
        with np.load(root / "evaluation/predictions.npz", allow_pickle=False) as z:
            data[arm] = {k: z[k].copy() for k in z.files}
        current = json.loads((root / "evaluation/samples.json").read_text())
        if samples is None:
            samples = current
        else:
            assert [
                (r["retained_index"], r["task"], r["source_frame_index"], r["source"]["root"])
                for r in current
            ] == [
                (r["retained_index"], r["task"], r["source_frame_index"], r["source"]["root"])
                for r in samples
            ]
    for skill, instruction in TASKS.items():
        indices = []
        for layout in ["B-layout1", "B-layout2"]:
            choices = [
                i
                for i, row in enumerate(samples)
                if row["task"] == instruction and row["lineage"]["layout"] == layout
            ]
            assert len(choices) >= 2
            indices.extend([choices[0], choices[len(choices) // 2]])
        images = []
        states = []
        seeds = []
        for index in indices:
            sample = samples[index]
            src = sample["source"]
            if src["root"] not in cache:
                cache[src["root"]] = LeRobotDataset(
                    src["repo_id"], root=Path(src["root"]), video_backend="pyav", return_uint8=True
                )
            item = cache[src["root"]][sample["source_frame_index"]]
            assert item["task"] == instruction
            state = item["observation.state"].numpy()
            assert np.array_equal(state, data["original"]["states"][index])
            images.append(item["observation.images.d435"].permute(1, 2, 0).numpy())
            states.append(state)
            seeds.append(42 + int(data["original"]["retained_indices"][index]))
        for arm in ["original", "flat"]:
            path = OUTPUT / (arm + "-" + skill + ".npz")
            np.savez_compressed(
                path,
                images=np.asarray(images),
                states=np.asarray(states),
                seeds=np.asarray(seeds),
                tasks=np.asarray([instruction] * len(indices)),
                expected_absolute_chunks=data[arm]["predictions"][indices],
            )
        rows.append(
            dict(
                skill=skill,
                samples=[
                    dict(
                        anchor=i,
                        retained_index=samples[i]["retained_index"],
                        source=samples[i]["source"]["root"],
                        source_frame=samples[i]["source_frame_index"],
                        layout=samples[i]["lineage"]["layout"],
                    )
                    for i in indices
                ],
            )
        )
    (OUTPUT / "reference-sources.json").write_text(
        json.dumps(
            dict(status="complete", new_inference_calls=0, models=["original", "flat"], tasks=rows),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                status="complete",
                output=str(OUTPUT),
                reference_files=8,
                recorded_RGB_frames=16,
                new_inference_calls=0,
            )
        )
    )


if __name__ == "__main__":
    main()
