# Native acceptance, 2026-11-01

Serving code: initial commit `8975852`. Hardware: one shared RTX 4090, GPU 2 on
the Ubuntu server. Existing localization serving remained running. Habitat-sim
0.2.4 is built headless with Bullet; the policy runs vLLM 0.19.1, Transformers
5.6.0 and Torch 2.10.0, BF16, eager execution, two concurrent episode sessions.

## Closed-loop checks

Each benchmark ran ten episodes, capped at three planning requests per episode.
These are diagnostic checks, not 500-action benchmark scores.

| Dataset | Valid plans | Executed actions | Mean policy RPC | Mean simulation RPC |
|---|---:|---:|---:|---:|
| R2R | 30 / 30 | 120 | 3.378s | 56.4ms |
| RxR English | 30 / 30 | 117 | 2.981s | 49.2ms |

Simulation RPC covers a prefix of up to four actual actions, immutable frame
storage and HTTP overhead. Policy time includes image fetching/preparation,
engine scheduling, prefill and generation; R2R includes the initial cold requests.
Image fetching/preparation averaged approximately 47ms per request. These short
mixed-history workloads do not establish a speedup for complete episodes.

The nav-train diagnostic adapter replayed all 237 executed actions without a
controller mismatch, including RxR's 30-degree turns. Separate real-Habitat checks
verified concurrent-session isolation, idempotent retries, stale-request rejection
and immutable historical RGB frames. See the [native simulation receipt](evidence/native-sim-acceptance.json)
and [vLLM acceptance receipt](evidence/vllm-acceptance.json).

## Input compatibility

Eight real reference images were processed through Swift's frozen training path
and the new policy path. All 2,392 token IDs, image grids and normalized pixel
values matched exactly. This required the original area-cap resize followed by
Qwen smart resize, with subsequent processor resizing disabled.

See [training input parity](evidence/prompt-parity.json). Six recorded live vLLM
requests, spanning one-, five- and eight-frame histories, also matched the
training-compatible processor's prompt token counts; see [live token checks](evidence/vllm-input-token-parity.json).
The live count check isolates tokenization/grid compatibility and does not compare
GPU encoder tensors.

## RxR navmesh correction

The original taller-agent navmesh made RxR episode 10035's start and goal fall
on different navigation islands. Rebuilding with a 1.5m agent kept it disconnected;
0.88m height and 0.18m radius restored a 7.922m path. The RxR service now uses its
own derived scene root, 30-degree turns and sliding disabled. Both camera profiles
retain the policy's trained 1.25m sensor height, so results use a diagnostic camera
protocol rather than the official challenge sensor configuration.

All 3,669 English RxR and all 1,839 R2R start/goal paths passed connectivity checks.
See [RxR connectivity](evidence/rxr-connectivity.json), [R2R connectivity](evidence/r2r-connectivity.json)
and [derived mesh provenance](evidence/rxr-navmesh-provenance.json).

## Regression and deployment

The independent engine has 13 passing CPU tests and a passing
[GitHub Actions run](https://github.com/AhernResearch/vln-sim-engine/actions/runs/41307036708).
The nav-train integration has 861 passing tests and 12 skipped tests.

Services listen on loopback: simulation 18810 and policy 18811. Deployment files,
full episode receipts and logs are under
`/home/admin1/nav_storage/outputs/sim_engine_20261101` on the server, shared as
`/mnt/navoutputs/sim_engine_20261101` on A100. The user units are
`nav-sim-server-20261101` and `nav-sim-policy-server-20261101`.

The separate 500-action camera cohorts run under
`nav-sim-vllm-camera-cohorts-20261101`: ten episodes each for R2R/RxR and each
camera profile. Their status is `cohort-status.json`, results are in `runs/`, and
paired reports are written to `comparisons/` after completion. No full-dataset
benchmark is started by that recipe.

Serving holds fixed checkpoint weights. Online weight updates and an RL learner
are not part of this acceptance.
