"""Event types and Event dataclass for the emergent-societies simulation."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Enumeration of all event types that can occur in the simulation."""

    SIM_START = "SIM_START"
    SIM_TICK = "SIM_TICK"
    MESSAGE = "MESSAGE"
    TRADE = "TRADE"
    COOPERATE = "COOPERATE"
    DEFECT = "DEFECT"
    RELATIONSHIP_UPDATE = "RELATIONSHIP_UPDATE"
    BELIEF_UPDATE = "BELIEF_UPDATE"


@dataclass
class Event:
    """Represents a single simulation event.

    Attributes:
        time: Simulation tick at which the event occurred.
        type: The kind of event (see :class:`EventType`).
        actor_id: Identifier of the agent that initiated the event.
        target_id: Identifier of the agent that is the target of the event,
            or ``None`` for events with no specific target.
        amount: Optional numeric value associated with the event (e.g. trade
            volume).
        payload: Arbitrary key-value data for event-specific metadata.
    """

    time: int
    type: EventType
    actor_id: str
    target_id: Optional[str] = None
    amount: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON/CSV-serialisable dictionary representation.

        Returns:
            A plain :class:`dict` with all fields, where :attr:`type` is
            replaced by its string value.
        """
        return {
            "time": self.time,
            "type": self.type.value,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "amount": self.amount,
            "payload": self.payload,
        }
