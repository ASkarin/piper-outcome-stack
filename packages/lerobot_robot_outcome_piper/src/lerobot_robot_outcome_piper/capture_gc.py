"""Defer automatic cyclic GC during timing-sensitive capture.

Reference counting continues normally. Collection runs before the episode begins,
and the prior GC setting is restored on every end/error/interrupt path.
Motion recorders retain deferral until after the control connection closes.
"""

import gc
import time


class CaptureGC:
    def __init__(self, collector=gc, clock=time.perf_counter):
        self.collector = collector
        self.clock = clock
        self.active = False
        self.was_enabled = None
        self.collection_ms = None
        self.started = False

    def prepare(self):
        if self.active:
            raise RuntimeError("cannot collect while sampling")
        self.started = False
        self.was_enabled = None
        if self.collector.isenabled():
            start = self.clock()
            self.collector.collect(2)
            self.collection_ms = (self.clock() - start) * 1000
        else:
            self.collection_ms = None

    def start(self):
        if self.active:
            raise RuntimeError("capture GC already deferred")
        self.was_enabled = self.collector.isenabled()
        self.active = True
        self.started = True
        if self.was_enabled:
            self.collector.disable()

    def stop(self):
        if self.active and self.was_enabled:
            self.collector.enable()
        self.active = False

    def summary(self):
        return dict(
            automatic_cyclic_gc_deferred=self.started,
            collection_before_episode_ms=self.collection_ms,
            prior_enabled=self.was_enabled,
            restored=not self.active,
        )
