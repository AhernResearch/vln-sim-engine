# Dual-backend validation — 2026-11-16

The shared sim-engine backend interface was validated against ordinary Habitat
0.3.1 and CUDA Habitat-GS 0.3.3 on A100 GPU 2. The independent source deployment
is `/data/nav_storage/sim-engine/deployments/gs-integration-20261116-v2`.
Existing production services were not restarted or reconfigured.

| Check | Result |
| --- | --- |
| Local Python 3.10 regression suite | 210 passed |
| Ruff (`src tests deployment`) | passed |
| Git whitespace check | passed |
| Wheel build and packaged license/backend files | passed; cp310-abi3 wheel |
| Wheel native extension import | passed on Python 3.10 and 3.11 |
| GS native render, fixed-state views, cached reset, native geometry queries/search | passed |
| Habitat native render, fixed-state views, cached reset, native geometry queries/search | passed |
| GS expert collection + recording validation | 13 steps, 0 collisions, 0.04147 m final goal distance |
| Habitat expert collection + recording validation | 9 steps, 0 collisions, 0.09335 m final goal distance |
| Both native HTTP probes | spawned worker, reset, action, deduplication, frozen historical depth, STOP and release passed |

The GS probe's 71 Python/native source and binary hashes match the local source
implementation. Both native probes exercise the same `deployment/probe_backend.py`.
The ordinary Habitat probe uses one training episode as an execution fixture;
neither probe runs a benchmark or admits training data.

[GS receipt](evidence/gs-integration-20261116/native-gs.json),
[Habitat receipt](evidence/gs-integration-20261116/native-habitat.json), and
[depth convention receipt](evidence/gs-integration-20261116/depth-contract.json)
contain the measurements and runtime identities. Earlier attempts and their
artifacts remain in the separate `gs-integration-20261116-v1` deployment.

Synthetic GS planes at 2 m and 4 m establish metric axial depth on binding
`03f2ae21f82c45abb0d667a41ac5cf486ba5f8feb34be8e53d00f89c8c3b9030`.
Maximum absolute errors were 0.00000191 m and 0.00005150 m respectively. This is a
projection/units test, not a validation of M-floor reconstruction accuracy.
The GS native probe intentionally permits raw capture without installing this
receipt as a visible-region planning approval. Production visible-region planning
requires the matching `simulation.depth_validation` report.

These native checks use scripted expert execution, with `model_inference=false`.
Real policy-model inference and the Harness migration were not exercised. The
existing model client and policy HTTP contract are unchanged. Collected GS
trajectories remain pending review, regardless of successful execution.

The host Python 3.10 interpreter lacks development headers. Building with the
installed Python 3.11 development headers produced the existing stable-ABI
`cp310-abi3` wheel, and the resulting native module imports on both interpreters.
The artifact is under `outputs/gs-integration-20261116/`.
