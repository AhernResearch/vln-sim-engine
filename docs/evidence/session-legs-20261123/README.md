# Planner-delegated legs: GPU acceptance (2026-11-23)

These checks run in-process on the A100 with the Habitat 0.3.1 environment (`sim-engine-031`)
on GPU 2, with no HTTP server and no model. `run.py` stands in for a planner: it picks a
visible safe floor pixel 1.5–3 m away in the view with the most such pixels. These are not
model decisions and not navigation scores. `config.json` is the exact settings file:

- R2R train from the harness data;
- camera `visible_88cm_hfov120`, 480×270;
- legs at their defaults (`snap_radius` 0.02, `same_level` false).

Source: branch `feat/production-kernel`. The uploaded source bundle's SHA-256 is
`c86f6ae1…6aee7`. The private rejection detail was added between the two runs.

## `routes/`: four routes, two legs, then a retrace

Each route runs observe → point_goal → observe → point_goal → retrace to leg 1 → score.

| Route | Result |
|---|---|
| 4966 | All legs `reached` |
| 3483 | All legs `reached` |
| 908 | All legs `reached` |
| 4324 | Both point goals `unreachable` |

On the three routes that reached:

- A point_goal leg takes 0.32–0.44 s for 14–20 control steps (about 23 ms per step,
  including panorama capture). observe takes 0.09–0.21 s.
- The grounded goal error is ≤ 7 cm.
- The retrace lands within 2.6 cm of the leg-1 end.
- Each leg starts exactly where the previous one ended.
- Public replies contain no private fields.

On 4324, the script's median-pixel pick chose a spot at the right image edge. Its smoothed
path failed the execution guard (`final surface path fails execution clearance`). The start
clearance is 0.22 m, and a comparable left-view goal plans successfully, so this is a
planner rejection of that spot, not a session fault.

A top-of-image pixel is `invalid_target` on every route.

## `cross-level/`: leg onto another level

| Route | Other-level safe pixels | Leg result |
|---|---|---|
| 4966 | 749 (front view) | `reached`, +0.59 m |
| 3483 | 32 (left view) | `reached`, +1.18 m |

The other-level pixels are connected safe floor pixels more than 0.2 m from the start
height. In each case the leg goes up the stairs.
