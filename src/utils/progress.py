"""
Progress bars that cannot take a training run down with them.

A 100-epoch run once died two hours in with `OSError: [Errno 22] Invalid
argument` raised from `tqdm -> print_status -> fp.write`. Epoch 1 finished
normally; the bar for epoch 2 then blocked writing to a stdout whose reader had
gone away, the run hung, and the handle eventually errored out. Two GPU hours
spent on a cosmetic write.

Two defences here:

* **Off when nobody is watching.** A progress bar is for an attached terminal.
  Detached runs - `schtasks`, `nohup`, a closed SSH session - get no bar at all,
  which is also what makes their logs readable. Per-epoch metrics still go to
  the log file, so nothing observable is lost.
* **Never fatal.** Should a write fail anyway, the bar is dropped and the run
  continues. Losing a progress bar is not a reason to lose a model.

`SESAML_PROGRESS=1` forces bars on, `0` forces them off, for when the
auto-detection guesses wrong.
"""

import os
import sys

from tqdm import tqdm as _tqdm


def progress_enabled() -> bool:
    """True when a progress bar has somewhere interactive to draw itself."""
    override = os.environ.get("SESAML_PROGRESS")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")

    # isatty() itself raises on some detached Windows handles.
    for stream in (sys.stderr, sys.stdout):
        try:
            if stream is not None and stream.isatty():
                return True
        except Exception:
            continue
    return False


class _SafeTqdm(_tqdm):
    """A tqdm whose display failures are swallowed rather than propagated."""

    def display(self, *args, **kwargs):
        try:
            return super().display(*args, **kwargs)
        except Exception:
            # Disable further drawing; the run carries on without a bar.
            self.disable = True
            return None

    def close(self, *args, **kwargs):
        try:
            return super().close(*args, **kwargs)
        except Exception:
            return None


def shutdown_loader_workers(*loaders) -> None:
    """
    Tears down a DataLoader's persistent worker processes.

    With `persistent_workers=True` the workers outlive each epoch by design, and
    are only reaped when the iterator is garbage collected. If the run raises
    while an iterator is still referenced - by a traceback frame, which is
    exactly when this matters - the workers are never collected, the parent
    blocks joining them at exit, and the whole tree sits holding GPU memory.
    One crashed run held 9.8GB for five hours that way.

    Best effort by design: this runs on the failure path, where raising a second
    exception would bury the first.
    """
    for loader in loaders:
        if loader is None:
            continue
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            continue
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
        try:
            loader._iterator = None
        except Exception:
            pass


def progress(iterable=None, **kwargs):
    """
    tqdm with the run-killing edge cases removed.

    `mininterval` is raised from tqdm's default 0.1s because a training loop
    redraws thousands of times per epoch, and each redraw is a write to a stream
    that may be a pipe.
    """
    kwargs.setdefault("disable", not progress_enabled())
    kwargs.setdefault("mininterval", 1.0)
    return _SafeTqdm(iterable, **kwargs)
