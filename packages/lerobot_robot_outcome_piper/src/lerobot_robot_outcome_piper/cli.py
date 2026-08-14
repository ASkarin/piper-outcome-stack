"""LeRobot CLI entry points with the required PiPER processor injected."""

import sys
from contextlib import contextmanager
from collections.abc import Iterator, Sequence

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.utils.import_utils import register_third_party_plugins

from .workflows import record, teleoperate


@parser.wrap()
def _teleoperate_from_cli(cfg: TeleoperateConfig) -> None:
    teleoperate(cfg)


@parser.wrap()
def _record_from_cli(cfg: RecordConfig):
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
    register_third_party_plugins()
    with _arguments(argv):
        _teleoperate_from_cli()


def record_main(argv: Sequence[str] | None = None):
    register_third_party_plugins()
    with _arguments(argv):
        return _record_from_cli()
