"""EventLogger: collects :class:`~simulation.events.Event` objects and writes them to JSONL or CSV."""

import csv
import json
from typing import Iterable, List

from simulation.events import Event


class EventLogger:
    """Accumulates simulation events and exports them to JSONL or CSV files.

    Attributes:
        events: Ordered list of logged :class:`~simulation.events.Event` objects.

    Example::

        logger = EventLogger()
        logger.log(Event(time=0, type=EventType.SIM_START, actor_id="sim"))
        logger.to_jsonl("/tmp/run.jsonl")
    """

    def __init__(self) -> None:
        """Initialise an empty EventLogger."""
        self.events: List[Event] = []

    def log(self, event: Event) -> None:
        """Append a single event to the log.

        Args:
            event: The :class:`~simulation.events.Event` to record.
        """
        self.events.append(event)

    def extend(self, events: Iterable[Event]) -> None:
        """Append multiple events to the log.

        Args:
            events: An iterable of :class:`~simulation.events.Event` objects.
        """
        self.events.extend(events)

    def to_jsonl(self, path: str) -> None:
        """Write all events to a JSON Lines file (one JSON object per line).

        Args:
            path: Filesystem path of the output ``.jsonl`` file.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(event.to_dict()) + "\n")

    def to_csv(self, path: str) -> None:
        """Write all events to a CSV file.

        The :attr:`~simulation.events.Event.payload` dict is serialised as a
        JSON string in a column named ``payload_json``.

        Args:
            path: Filesystem path of the output ``.csv`` file.
        """
        fieldnames = ["time", "type", "actor_id", "target_id", "amount", "payload_json"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.events:
                row = event.to_dict()
                row["payload_json"] = json.dumps(row.pop("payload"))
                writer.writerow(row)

    def clear(self) -> None:
        """Remove all events from the log."""
        self.events.clear()
