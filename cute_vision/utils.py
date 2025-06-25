from typing import Any


def first(iterable: Any) -> Any:
    return next(iter(iterable))


def safe_first(iterable: Any, default: Any = None) -> Any | None:
    if iterable is None:
        return default
    try:
        return first(iterable)
    except StopIteration:
        return default
