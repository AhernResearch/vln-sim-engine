"""Runtime selection occurs only inside the owning worker process."""
from .contracts import Capture, Environment


def create_environment(settings):
    from .habitat import GaussianEnvironment, HabitatEnvironment

    adapter = GaussianEnvironment if settings.simulation.backend == "habitat_gs" else HabitatEnvironment
    return adapter(settings)


__all__ = ["Capture", "Environment", "create_environment"]
