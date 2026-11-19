"""Pinned simulator runtimes. A runtime decides the binding version, renderer requirements,
which scene sources it can load and where its depth comes from; nothing else branches on it."""

from __future__ import annotations

from dataclasses import dataclass

MESH_SCENES = "mesh"
SCAN_PACKAGES = "scan_package"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    version: str
    scene_source: str
    depth_source: str
    requires_cuda: bool = False
    # A binding attribute that only the pinned build exposes.
    marker: str | None = None
    physics: bool = True

    def validate(self, hs):
        if hs.__version__ != self.version or (self.requires_cuda and not hs.cuda_enabled):
            raise RuntimeError(f"{self.name} {self.version}"
                               + (" with CUDA" if self.requires_cuda else "") + " is required")
        if self.marker and not hasattr(hs.SimulatorConfiguration(), self.marker):
            raise RuntimeError(f"pinned {self.name} bindings are required")

    def configure(self, config):
        if not self.physics:
            config.enable_physics = False
            config.navmesh_settings = None


HABITAT = RuntimeProfile("habitat", "0.3.1", MESH_SCENES, "mesh")
HABITAT_GS = RuntimeProfile("habitat_gs", "0.3.3", SCAN_PACKAGES, "gaussian_unvalidated",
                            requires_cuda=True, marker="gaussian_time", physics=False)
RUNTIMES = {profile.name: profile for profile in (HABITAT, HABITAT_GS)}


def runtime(name: str) -> RuntimeProfile:
    return RUNTIMES[name]
