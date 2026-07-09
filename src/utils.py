"""Small helpers shared across the SphereMeshTool package."""

from __future__ import annotations

import time
from typing import Any, Callable


def measure_time(
    fn: Callable[..., Any],
    *args: Any,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` with the given arguments, print elapsed wall time, return its result.

    Use ``measure_time(foo, n, a, mode=True)`` so the timed call is ``foo(n, a, mode=True)``.

    Do **not** write ``measure_time(foo(n, a, mode=True))``: that calls ``foo`` first and
    only passes its return value into ``measure_time``, so timing would be meaningless.

    For expressions or one-off calls with no separate argument list, use a zero-argument
    callable, e.g. ``measure_time(lambda: foo(n, a, mode=True), label="foo")`` so the log
    does not show ``<lambda>``.

    Prints a line like ``Time for foo: 0.000020 s``. If *label* is given, it is used in
    the message; otherwise the name comes from ``fn.__name__``.
    """
    display = label if label is not None else getattr(fn, "__name__", type(fn).__name__)
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        print(f"Time for {display}: {elapsed:.6f} s")
