"""Bounded in-memory diagnostics; serialize only after the control session closes."""

from copy import deepcopy
import json
from pathlib import Path


class ControlTrace:
    def __init__(self, path, max_events=20000):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")
        self.events = []
        self.max_events = max_events
        self.dropped = 0

    def append(self, event):
        # A long diagnostic run must not grow memory indefinitely or change motion.
        # Overflow is explicit in the final summary; it is not a complete trace.
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return
        self.events.append(deepcopy(event))

    def save(self, **summary):
        try:
            for event in self.events:
                self.stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.stream.write(
                json.dumps(
                    dict(
                        event="summary",
                        events=len(self.events),
                        dropped_events=self.dropped,
                        trace_complete=self.dropped == 0,
                        **summary,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
        finally:
            self.stream.close()


class PreparationTimingTrace:
    """Preparation observations/actions go to existing bounded telemetry, never Dataset rows."""

    def __init__(self, emit):
        self.emit = emit
        self.observation = None

    def append(self, event):
        if event["event"] == "observation":
            self.observation = deepcopy(event)
        elif event["event"] == "action":
            self.emit(
                "preparation_tick",
                observation=self.observation,
                **{k: v for k, v in event.items() if k != "event"},
            )
