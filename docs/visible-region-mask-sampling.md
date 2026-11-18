# Visible-floor candidate sampling

The free-task collection service now selects candidate pixels directly from the
intersection of the original model box and its visible-safe-floor mask. A fixed
uniform grid could miss all support in a tiny patch even when the mask contained
valid pixels. No box expansion, prompt change or movement retry is involved.

`navigation.visual_region.sample_region` keeps the existing geometric acceptance
checks and center/clearance candidate ranking. With a mask, pixel selection is:

1. Keep only safe pixel centers inside the original normalized box.
2. If there are at most `grid_size**2` pixels, check every one.
3. Otherwise partition the box into `grid_size × grid_size` cells. Take a safe
   pixel nearest each occupied cell's center, then fill remaining slots with
   pixels nearest the box center. Selection and tie breaking are deterministic.
4. Backproject, snap and check each selected pixel as before: current floor,
   navigability, clearance, reprojection inside the original box, connectivity and
   configured distance limits. Deduplicate goals at the existing 2.5 cm resolution.
5. Pass candidates to the existing bounded planner, local refinement and swept
   controller preview. Only a feasible selection can reach actual execution.

At the current `grid_size=13`, at most 169 pixels receive geometric/geodesic checks.
The planning limit remains 12 candidates and 64 total original/adjusted previews.
There is no 8 m collection cap. Small support patches within the pixel budget no
longer disappear solely because a uniform grid misses their pixel coordinates;
dense masks still have bounded coverage and do not guarantee every feasible goal
will be examined. Without a mask, the historical uniform-grid behavior remains.

`region-candidates.json` records `sampling_policy`, `sampling_budget`,
`safe_pixels_in_bbox`, `sampling_cells_occupied`, actual `samples`, candidate count
and rejection reasons. The production policy is `visible_safe_mask_stratified_v1`.
`grid_size` now also defines spatial coverage and the sample budget when a mask is
present. `samples` is the number actually checked, rather than always 169.

## Fixed-input native regression

The 16 saved free-task boxes and RGB-D observations from the prior concurrent test
were replayed through read-only NavMesh planning and controller preview. No model
requests or physical movement were issued. All 13 formerly feasible cases remained
feasible; the other three remained infeasible.

The reported `loc01-right` patch contained six safe pixels. The old uniform grid
produced zero candidates. The new sampler checked all six, produced four distinct
goals, and performed eight bounded original/adjusted planning attempts: seven
failed swept execution clearance in preview; one failed final path clearance.
The sampling miss is fixed, but this particular box still cannot be admitted for
movement under the current checks. It is not counted as a recovered trajectory.
The two boxes with no safe support still produced zero samples and no movement.

## Fresh end-to-end regression

A separate run used 16 fresh model proposals through the normal nav-train client
and the updated live 16-worker service. It completed 12 trajectories and rejected
four before movement, with no model failures, uncertain executions or collisions.
All 249 RGB-D/state/action frames and PointNav frame labels were checked, including
independent first/last-frame coordinate transforms and pre-movement target records.
Wall time was 32.08 s. Proposals differed from the fixed-input regression; these
counts are not a paired yield or throughput comparison.

In the fresh `loc01-right` proposal the model changed the box from
`[350,750,650,950]` to `[350,850,650,1000]`. All six safe pixels were sampled, but
all snapped points reprojected outside that tighter original box. It was correctly
rejected before movement. The other three fresh exclusions had no safe support.
No completed attempts or old evidence were overwritten. All outputs remain
`audit_only` / `pending_review` and no training export was made.

## Deployment and checks

Current frozen deployment:
`/data/nav_storage/sim-engine/deployments/collection-mask-sampling-36c9551ce98f`.
Service owner: `29527adbc69e4b92a8168b88bbc7874d`. Port 18812, root `train-v1`,
16 workers, one cached scene per worker and queue capacity 128 are unchanged. Only
the owned idle previous service was stopped, after owner/PID/start-time/configuration
checks. The service ended healthy with 16 live workers and zero active tasks.

Sim-engine: 105 tests passed, including 12 new sparse/dense-mask, bounds and geometric
guard tests; Ruff passed. Nav-train client tests: 4 passed. Required harness checks
also passed: 311 tests plus 4 subtests, Ruff, mypy, undefined-name checks and frontend
build. No frontend workflow changed.

- [Fixed-input read-only results](evidence/collection-mask-sampling-20261114.readonly.json)
- [Fresh end-to-end results](evidence/collection-mask-sampling-20261114.e2e.json)
- [Service source identity](evidence/collection-mask-sampling-20261114.health.json)

Read-only evidence root:
`/data/nav_storage/sim-engine/collections/mask-sampling-36c9551ce98f`.
Fresh collection root:
`/data/nav_storage/nav-train/collections/mask-sampling-e2e-20261114-6edf0c6d798d`.
