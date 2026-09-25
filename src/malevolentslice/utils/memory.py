import gc
import psutil
import os

def force_garbage_collection() -> None:
    """Explicitly trigger Python garbage collection to free unreferenced arrays."""
    gc.collect()

def get_memory_usage_mb() -> float:
    """
    Returns the current process memory consumption in MB.
    On macOS (Darwin), RSS artificially includes shared system dynamic library
    caches (Accelerate framework, dyld cache, etc.) which can inflate the
    reported metric by 300-400 MB. We use 'phys_footprint' on macOS, which is
    Apple's official and accurate metric for actual physical RAM allocation.
    On Windows and Linux, RSS accurately represents the process Working Set / VmRSS.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    if hasattr(mem_info, "phys_footprint"):
        return mem_info.phys_footprint / (1024.0 * 1024.0)
    return mem_info.rss / (1024.0 * 1024.0)

class MemoryTracker:
    """Context manager and tracker for monitoring peak RAM consumption and memory growth."""
    def __init__(self):
        self.start_memory_mb = get_memory_usage_mb()
        self.peak_memory_mb = self.start_memory_mb

    def update(self) -> float:
        """Update and return the peak RSS memory in MB."""
        current = get_memory_usage_mb()
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current
        return current

    def get_peak_mb(self) -> float:
        self.update()
        return self.peak_memory_mb

    def get_growth_mb(self) -> float:
        """Returns the net memory growth (peak RSS - initial RSS) in MB."""
        self.update()
        return max(0.0, self.peak_memory_mb - self.start_memory_mb)
