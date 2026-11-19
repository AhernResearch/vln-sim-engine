"""Runtime selection occurs only inside the owning worker process."""
from .contracts import Capture, Environment


def create_environment(settings):
    from .habitat import HabitatEnvironment

    return HabitatEnvironment(settings)


__all__ = ["Capture", "Environment", "create_environment"]
