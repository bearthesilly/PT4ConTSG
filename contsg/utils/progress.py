"""
Progress utilities for ConTSG.

Provides smart progress reporting that automatically adapts to the environment:
- Interactive terminals: Use tqdm progress bars
- Non-interactive (Slurm, pipes): Use simple log output
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


class ProgressMode(str, Enum):
    """Progress display modes."""
    AUTO = "auto"      # Auto-detect based on TTY
    TQDM = "tqdm"      # Force tqdm progress bar
    LOG = "log"        # Force log-style output
    OFF = "off"        # Disable progress entirely


# Global progress mode (can be set by CLI)
_progress_mode: ProgressMode = ProgressMode.AUTO


def set_progress_mode(mode: ProgressMode | str) -> None:
    """Set global progress mode."""
    global _progress_mode
    if isinstance(mode, str):
        mode = ProgressMode(mode)
    _progress_mode = mode


def get_progress_mode() -> ProgressMode:
    """Get current progress mode."""
    return _progress_mode


def is_interactive() -> bool:
    """Check if running in an interactive terminal."""
    try:
        return sys.stdout.isatty() and sys.stderr.isatty()
    except Exception:
        return False


def _should_use_tqdm() -> bool:
    """Determine whether to use tqdm based on mode and environment."""
    mode = get_progress_mode()

    if mode == ProgressMode.TQDM:
        return True
    elif mode == ProgressMode.LOG:
        return False
    elif mode == ProgressMode.OFF:
        return False
    else:  # AUTO
        return is_interactive()


class LogProgress:
    """
    Simple log-based progress reporter for non-interactive environments.

    Instead of updating a progress bar, prints periodic status messages.
    """

    def __init__(
        self,
        iterable: Iterable[T],
        desc: str = "",
        total: int | None = None,
        log_interval: float = 0.1,  # Log every 10% by default
        disable: bool = False,
    ):
        self.iterable = iterable
        self.desc = desc
        self.disable = disable
        self.log_interval = log_interval

        # Get total if not provided
        if total is None:
            try:
                self.total = len(iterable)  # type: ignore
            except TypeError:
                self.total = None
        else:
            self.total = total

        self.n = 0
        self._last_logged_pct = -1

    def __iter__(self) -> Iterator[T]:
        if self.disable:
            yield from self.iterable
            return

        # Print start message
        if self.total:
            print(f"[{self.desc}] Starting (0/{self.total})...", flush=True)
        else:
            print(f"[{self.desc}] Starting...", flush=True)

        for item in self.iterable:
            yield item
            self.n += 1
            self._maybe_log()

        # Print completion message
        print(f"[{self.desc}] Completed ({self.n} items)", flush=True)

    def _maybe_log(self) -> None:
        """Log progress at intervals."""
        if self.total is None or self.total == 0:
            return

        pct = self.n / self.total
        # Check if we crossed a log interval boundary
        current_interval = int(pct / self.log_interval)
        if current_interval > self._last_logged_pct:
            self._last_logged_pct = current_interval
            print(f"[{self.desc}] {self.n}/{self.total} ({pct:.0%})", flush=True)

    def __len__(self) -> int:
        if self.total is not None:
            return self.total
        raise TypeError("Length unknown")


def smart_tqdm(
    iterable: Iterable[T],
    desc: str = "",
    total: int | None = None,
    disable: bool = False,
    log_interval: float = 0.1,
    **tqdm_kwargs,
) -> Iterable[T]:
    """
    Smart progress wrapper that adapts to the environment.

    In interactive terminals, uses tqdm for a progress bar.
    In non-interactive environments (Slurm, pipes), uses simple log output.

    Args:
        iterable: The iterable to wrap
        desc: Description for the progress bar/log
        total: Total number of items (optional, auto-detected if possible)
        disable: Force disable progress output
        log_interval: For log mode, how often to log (0.1 = every 10%)
        **tqdm_kwargs: Additional arguments passed to tqdm

    Returns:
        Wrapped iterable

    Example:
        for batch in smart_tqdm(dataloader, desc="Training"):
            process(batch)
    """
    mode = get_progress_mode()

    if disable or mode == ProgressMode.OFF:
        return iterable

    if _should_use_tqdm():
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, **tqdm_kwargs)
    else:
        return LogProgress(
            iterable,
            desc=desc,
            total=total,
            log_interval=log_interval,
            disable=disable,
        )
