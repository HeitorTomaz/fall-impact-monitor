"""Events: model, categories and near-miss classification."""

from fall_impact_monitor.events.model import Event, EventCategory, is_near_miss, make_event

__all__ = ["Event", "EventCategory", "is_near_miss", "make_event"]
