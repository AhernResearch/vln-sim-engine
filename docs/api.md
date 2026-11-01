# HTTP API

The simulator defaults to loopback port 18810; policy serving uses 18811.

| Method | Endpoint | Meaning |
|---|---|---|
| GET | `/v1/health` | Service identity and protocol configuration |
| GET | `/v1/datasets` | Registered annotation identities and episode counts |
| POST | `/v1/sessions` | Reserve a worker and reset an episode |
| POST | `/v1/sessions/{id}/step` | Execute a short local trajectory prefix |
| GET | `/v1/sessions/{id}/frames/{event}` | Immutable packed RGB8 bytes and hash headers |
| GET | `/v1/sessions/{id}/metrics` | Privileged metrics and execution trace |
| POST | `/v1/sessions/{id}/finish` | Explicit stop, model failure or cancellation |
| DELETE | `/v1/sessions/{id}` | Release the episode and worker |
| POST | `/v1/policy/generate` | Asynchronous vLLM completion; policy service only |

Reset:

```json
{"dataset":"r2r_val_unseen","episode_id":"10","camera":"trained_ce30","seed":0}
```

The response contains `session_id`, instruction, camera metadata, current event
index and observation handles. It contains no goal/reference/global pose.

Policy generation:

```json
{"request_id":"unique-generation-id","instruction":"Walk through the doorway.",
 "camera":"trained_ce30","observations":[{"session_id":"returned-session-id","event_index":0}],
 "temperature":0,"seed":0,"max_tokens":512}
```

The result carries `text`, `valid`, `trajectory`, token IDs, sampled-token log
probabilities, policy version and timing. Evaluation uses temperature 0. RL
callers must record sampling parameters and verify log-probability semantics for
their training algorithm. Reusing a request ID with different inputs is rejected.

Execution:

```json
{"request_id":"unique-step-id","expected_event_index":0,"action_budget":4,
 "policy_version":"qwen3-vl-4b-epoch1",
 "trajectory":[[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],
               [1.25,0,0],[1.5,0,0],[1.75,0,0],[2,0,0]]}
```

Execution returns actual actions, per-action observation handles, collision count,
new event index and termination. Optional training rewards are separate fields and
must not be projected into policy requests. All history observations must belong
to one active session and have unique increasing event indices.

Frame bytes use shape `(height,width,3)`, uint8 RGB. Headers provide `X-Width`,
`X-Height`, `X-Camera-Profile` and `X-Content-SHA256`. Frames are valid for the
session's lifetime; result writers must fetch any required replay media before
closing it.

Unknown identifiers return 404; invalid schemas return 422; stale/conflicting
steps return 409; occupied simulation capacity or a full policy queue returns
429. Numerical, worker or inference failures propagate as 5xx, without placeholder
scores. Clients must not retry execution with a new ID after an uncertain outcome.

Physics and asset identities are dataset-specific: simulator health exposes
`protocol.dataset_physics[dataset]` and `protocol.scene_hashes[dataset]`. This
permits different agent dimensions, turn angles and navmeshes on the same worker
pool. Dataset identity and physics must be checked before comparing runs.
