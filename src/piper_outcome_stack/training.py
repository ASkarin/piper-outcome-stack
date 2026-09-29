"""Select policy observations, then delegate training unchanged to official LeRobot."""


def select_policy_inputs(features, image_keys=None):
    from lerobot.utils.feature_utils import dataset_to_policy_features

    converted = dataset_to_policy_features(features)
    if image_keys is None:
        image_keys = [
            key
            for key, ft in features.items()
            if ft["dtype"] in ("image", "video")
            and not (ft.get("info") or {}).get("is_depth_map", False)
        ]
    result = {"observation.state": converted["observation.state"]}
    for key in image_keys:
        if key not in features or features[key]["dtype"] not in ("image", "video"):
            raise ValueError(f"not an available image feature: {key}")
        result[key] = converted[key]
    return result


def train_main(argv=None):
    import sys
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.scripts.lerobot_train import train

    @parser.wrap()
    def run(cfg: TrainPipelineConfig):
        from pathlib import Path

        if (
            cfg.dataset.root is not None
            and (Path(cfg.dataset.root) / "xbox-conversion.json").exists()
        ):
            from .training_selection import require_successful_selection

            require_successful_selection(cfg.dataset.root, cfg.dataset.episodes)
        if (
            cfg.policy is not None
            and not cfg.policy.input_features
            and not cfg.resume
            and cfg.policy.pretrained_path is None
            and parser.get_path_arg("policy") is None
            and parser.get_path_arg("reward_model") is None
        ):
            meta = LeRobotDatasetMetadata(
                cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
            )
            cfg.policy.input_features = select_policy_inputs(meta.features)
        # Explicit checkpoint/experiment inputs are never silently overwritten.
        return train(cfg)

    original = sys.argv
    try:
        if argv is not None:
            sys.argv = [original[0], *argv]
        return run()
    finally:
        sys.argv = original
