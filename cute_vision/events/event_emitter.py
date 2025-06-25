from typing import Any
from pydispatch import dispatcher


class EventEmitter:

    def __init__(self, event_name: str):
        self.event_name = event_name

    def __call__(self, emit: bool, passthrough: Any = None, event_kwargs: dict | None = None):

        if event_kwargs is None:
            event_kwargs = {}

        if emit:
            dispatcher.send(signal=self.event_name, sender=self, **event_kwargs)

        return passthrough
