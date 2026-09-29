"""Official child writer lifecycle; fake numeric/image data, no device access."""

import json
import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_outcome_piper.episode_save import compress_episode_depth


@pytest.mark.parametrize("kind", ["npz", "tiff"])
def test_compression_verification_failure_preserves_original(tmp_path, monkeypatch, kind):
    from PIL import Image
    from lerobot_robot_outcome_piper import episode_save

    path = tmp_path / ("depth." + kind)
    if kind == "npz":
        np.savez(path, depth=np.arange(64, dtype=np.uint16).reshape(8, 8))
    else:
        Image.fromarray(np.arange(64, dtype=np.float32).reshape(8, 8)).save(path)
    original = path.read_bytes()
    monkeypatch.setattr(episode_save, "_same_array", lambda *args: False)
    with pytest.raises(RuntimeError, match="compression verification failed"):
        episode_save.compress_depth_file(path, kind)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp"))  # The rejected candidate is diagnostic evidence.


def test_compressed_depth_is_embedded_and_raw_npz_still_loads(tmp_path, capsys):
    import io
    import zipfile
    import pyarrow.parquet as pq
    from PIL import Image

    root = tmp_path / "dataset"
    key = "observation.images.depth"
    ds = LeRobotDataset.create(
        "local/compressed-depth",
        20,
        root=root,
        use_videos=False,
        features={
            key: {
                "dtype": "image",
                "shape": (32, 32, 1),
                "names": ["h", "w", "c"],
                "info": {"is_depth_map": True, "depth_unit": "m"},
            }
        },
    )
    raw = np.zeros((32, 32), dtype=np.uint16)
    raw[0, :4] = [0, 1, 65534, 65535]
    metric = raw.astype(np.float32) * np.float32(0.001)
    path = root / "raw.npz"
    np.savez(path, depth=raw)
    try:
        ds.add_frame({key: metric[..., None], "task": "compression roundtrip"})
        ds.writer._wait_image_writer()
        compress_episode_depth(
            ds.writer.episode_buffer, ds.features, [path], tmp_path / "compression.json"
        )
        ds.save_episode()
        ds.finalize()
        with np.load(path) as values:
            np.testing.assert_array_equal(values["depth"], raw)
            assert values["depth"].dtype == raw.dtype
        with zipfile.ZipFile(path) as archive:
            assert archive.getinfo("depth.npy").compress_type == zipfile.ZIP_DEFLATED
        encoded = pq.read_table(next((root / "data").rglob("*.parquet"))).to_pylist()[0][key][
            "bytes"
        ]
        with Image.open(io.BytesIO(encoded)) as restored:
            assert restored.tag_v2[259] in (8, 32946)  # TIFF Deflate, not raw TIFF.
            assert np.asarray(restored).dtype == np.float32
            np.testing.assert_array_equal(np.asarray(restored), metric)
        loaded = LeRobotDataset(ds.repo_id, root=root, depth_output_unit="m")
        np.testing.assert_array_equal(loaded[0][key].numpy().squeeze(0), metric)
        report = json.loads((tmp_path / "compression.json").read_text())
        assert report["status"] == "verified"
        assert report["compression_level"] == 1
        output = capsys.readouterr().out
        assert "深度NPZ" in output and "深度TIFF" in output
        assert "NPZ" in output and "TIFF" in output
        assert report["npz"]["files"] == report["tiff"]["files"] == 1
    finally:
        ds.finalize()


def test_compression_write_failure_preserves_source_and_records_error(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper.episode_save import compress_episode_depth

    path = tmp_path / "depth.npz"
    np.savez(path, depth=np.zeros((8, 8), dtype=np.uint16))
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(np.lib.format, "write_array", fail)
    report = tmp_path / "compression.json"
    with pytest.raises(OSError, match="disk write failed"):
        compress_episode_depth({}, {}, [path], report)
    assert path.read_bytes() == original
    result = json.loads(report.read_text())
    assert result["status"] == "failed"
    assert result["path"] == str(path)
