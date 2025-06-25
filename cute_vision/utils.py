from typing import Any


def first(iterable: Any) -> Any:
    return next(iter(iterable))


def safe_first(iterable: Any) -> Any | None:
    if iterable is None:
        return None
    try:
        return first(iterable)
    except StopIteration:
        return None
