import gc
import time

from loguru import logger


class GarbageCollector:
    def __init__(
        self,
        gc_freq: int | None,
    ) -> None:
        assert gc_freq is None or gc_freq > 0, "gc_freq must be positive or None"
        self.gc_freq = gc_freq
        if self.gc_freq is not None:
            gc.disable()

    def collect(
        self,
        step: int,
    ) -> None:
        if self.gc_freq is None:
            return

        if step % self.gc_freq == 0:
            begin = time.perf_counter()
            gc.collect(generation=1)
            end = time.perf_counter()
            logger.debug(f"Garbage collection at step {step} took {end - begin:.4f} seconds")
