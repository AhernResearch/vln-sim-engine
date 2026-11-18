"""Compatibility exports for existing integrations."""
from .backends.habitat import HABITAT_VERSION
from .backends.habitat import HabitatEnvironment as _HabitatEnvironment
from .tasks import metrics


class HabitatEnvironment(_HabitatEnvironment):
    metrics = metrics


__all__ = ["HABITAT_VERSION", "HabitatEnvironment"]
