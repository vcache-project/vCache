import os
from typing import Optional

import psutil

_gpu_unavailable: bool = False
_token_encoding = None
_tiktoken_unavailable: bool = False


class ResourceSampler:
    """Samples CPU and memory usage of the current process."""

    def __init__(self):
        self._process: psutil.Process = psutil.Process(os.getpid())
        # Priming call: psutil.cpu_percent() always returns 0.0 on the first
        # invocation, since it measures usage since the *previous* call.
        self._process.cpu_percent(interval=None)

    def cpu_percent(self) -> float:
        """Returns process CPU usage in percent since the last sample."""
        return self._process.cpu_percent(interval=None)

    def memory_mb(self) -> float:
        """Returns the process's resident set size (RSS) in megabytes."""
        return self._process.memory_info().rss / (1024 * 1024)


def gpu_utilization_percent() -> Optional[float]:
    """Returns the current GPU utilization in percent, if available.

    This is best-effort: it returns None if `pynvml` isn't installed, no
    NVIDIA driver is present, or querying the device otherwise fails. Once
    unavailable, it is assumed to stay unavailable for the process lifetime
    so we don't re-probe on every call.

    Returns:
        The utilization of GPU device 0 in percent, or None if unavailable.
    """
    global _gpu_unavailable
    if _gpu_unavailable:
        return None

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        _gpu_unavailable = True
        return None


def count_tokens(text: str) -> int:
    """Estimates the number of tokens in `text`.

    Uses tiktoken's cl100k_base encoding when available, falling back to a
    whitespace-based word count otherwise.

    Args:
        text: The text to count tokens for.

    Returns:
        The estimated token count. 0 for empty/None text.
    """
    global _token_encoding, _tiktoken_unavailable
    if not text:
        return 0

    if not _tiktoken_unavailable and _token_encoding is None:
        try:
            import tiktoken

            _token_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_unavailable = True

    if _token_encoding is not None:
        return len(_token_encoding.encode(text))
    return len(text.split())
