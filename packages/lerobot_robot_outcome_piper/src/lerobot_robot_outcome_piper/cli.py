"""LeRobot CLI entry points with the required PiPER processor injected."""

import sys
from dataclasses import dataclass
from contextlib import contextmanager
from collections.abc import Iterator, Sequence

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.utils.import_utils import register_third_party_plugins

from .workflows import record, teleoperate, replay
from .console import operator_console
from lerobot.scripts.lerobot_replay import ReplayConfig


@dataclass
class PiperTeleoperateConfig(TeleoperateConfig):
    control_trace_path: str | None = None


@dataclass
class PiperRecordConfig(RecordConfig):
    """Depth capture is independent of optional metric depth Dataset columns."""

    export_depth: bool = False
    raw_root: str | None = None


@dataclass
class PiperReplayConfig(ReplayConfig):
    trajectory: dict | None = None
    trajectory_report_path: str | None = None


@parser.wrap()
def _teleoperate_from_cli(cfg: PiperTeleoperateConfig) -> None:
    teleoperate(cfg)


@parser.wrap()
def _record_from_cli(cfg: PiperRecordConfig):
    return record(cfg)


@contextmanager
def _arguments(argv: Sequence[str] | None) -> Iterator[None]:
    if argv is None:
        yield
        return
    original = sys.argv
    sys.argv = [original[0], *argv]
    try:
        yield
    finally:
        sys.argv = original


def teleoperate_main(argv: Sequence[str] | None = None) -> None:
    with _arguments(argv), operator_console(sys.argv[1:]):
        register_third_party_plugins()
        _teleoperate_from_cli()


def record_main(argv: Sequence[str] | None = None):
    with _arguments(argv), operator_console(sys.argv[1:]):
        register_third_party_plugins()
        return _record_from_cli()


@parser.wrap()
def _replay_from_cli(cfg: PiperReplayConfig) -> None:
    replay(cfg)


def replay_main(argv: Sequence[str] | None = None) -> None:
    with _arguments(argv), operator_console(sys.argv[1:]):
        register_third_party_plugins()
        _replay_from_cli()
