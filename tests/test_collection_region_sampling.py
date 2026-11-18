"""Sparse visible support must survive sampling without bypassing target checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim_engine.collection.visual_region import inside_bbox, sample_region


def scene():
    depth = np.zeros((101, 101), dtype=np.float32)
    for y in range(51, 101):
        depth[y, :] = 50 / (y - 50)
    return dict(
        bbox=[400, 700, 700, 900],
        depth=depth,
        intrinsics=[[50, 0, 50], [0, 50, 50], [0, 0, 1]],
        sensor={"position": [0, 1, 0], "rotation_xyzw": [0, 0, 0, 1]},
        start=[0, 0, 0],
        mesh=SimpleNamespace(snap_point=lambda p: p, is_navigable=lambda p: True),
        guard=SimpleNamespace(clearance=lambda p: 0.1, profile=SimpleNamespace(clearance_m=0.05)),
        shortest=lambda a, b: [a, b],
    )


def test_single_safe_pixel_between_grid_lines_is_grounded_inside_original_box():
    args = scene()
    mask = np.zeros(args["depth"].shape, dtype=bool)
    mask[74, 54] = True
    # This exact support is absent from the former 13x13 uniform sampling grid.
    uniform = sample_region(**args)
    assert [54, 74] not in [c["pixel_xy"] for c in uniform["candidates"]]
    result = sample_region(**args, safe_mask=mask)
    assert result["samples"] == result["candidate_count"] == result["safe_pixels_in_bbox"] == 1
    assert result["candidates"][0]["pixel_xy"] == [54, 74]
    assert result["box_expanded"] is False
    assert inside_bbox(result["candidates"][0]["projected_uv"], args["bbox"])


def test_all_six_sparse_support_pixels_are_checked_even_in_one_sampling_cell():
    args = scene()
    mask = np.zeros(args["depth"].shape, dtype=bool)
    pixels = [(x, y) for y in (71, 72) for x in (48, 49, 50)]
    for x, y in pixels:
        mask[y, x] = True
    result = sample_region(**args, grid_size=5, safe_mask=mask)
    assert result["samples"] == 6
    assert result["sampling_cells_occupied"] == 1
    assert {tuple(c["pixel_xy"]) for c in result["candidates"]} == set(pixels)
    assert result["sampling_policy"] == "visible_safe_mask_stratified_v1"


def test_dense_support_keeps_an_isolated_patch_and_respects_original_geometry_budget():
    args = scene()
    mask = np.zeros(args["depth"].shape, dtype=bool)
    mask[75:91, 45:65] = True
    mask[71, 69] = True
    calls = []

    def shortest(a, b):
        calls.append(b)
        return [a, b]

    args["shortest"] = shortest
    result = sample_region(**args, grid_size=5, safe_mask=mask)
    assert result["samples"] == 25
    assert len(calls) <= 25
    assert [69, 71] in [c["pixel_xy"] for c in result["candidates"]]
    assert all(mask[c["pixel_xy"][1], c["pixel_xy"][0]] for c in result["candidates"])
    assert all(inside_bbox(c["sample_uv"], args["bbox"]) for c in result["candidates"])
    assert result == sample_region(**args, grid_size=5, safe_mask=mask)


@pytest.mark.parametrize("support_elsewhere", [False, True])
def test_empty_box_support_never_falls_back_outside_the_box(support_elsewhere):
    args = scene()
    mask = np.zeros(args["depth"].shape, dtype=bool)
    if support_elsewhere:
        mask[92:, :] = True

    def unexpected(*args):
        pytest.fail("empty original box must not invoke geometry checks")

    args["shortest"] = unexpected
    result = sample_region(**args, safe_mask=mask)
    assert result["safe_pixels_in_bbox"] == result["samples"] == result["candidate_count"] == 0
    assert result["box_expanded"] is False


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("clearance", "insufficient endpoint clearance"),
        ("navigability", "not navigable"),
        ("height", "selected pixel is not near navigable current-floor surface"),
        ("box", "projected NavMesh point leaves original box"),
        ("route", "disconnected"),
        ("depth", "selected point has invalid depth"),
    ],
)
def test_mask_samples_still_require_depth_surface_clearance_box_and_route_checks(problem, reason):
    args = scene()
    mask = np.zeros(args["depth"].shape, dtype=bool)
    mask[74, 69] = True
    if problem == "clearance":
        args["guard"].clearance = lambda p: 0.01
    elif problem == "navigability":
        args["mesh"].is_navigable = lambda p: False
    elif problem == "height":
        args["mesh"].snap_point = lambda p: p + [0, 0.21, 0]
    elif problem == "box":
        args["mesh"].snap_point = lambda p: p + [0.049, 0, 0]
    elif problem == "route":

        def disconnected(a, b):
            raise ValueError("disconnected")

        args["shortest"] = disconnected
    else:
        args["depth"][74, 69] = np.nan
    result = sample_region(**args, safe_mask=mask)
    assert result["candidate_count"] == 0
    assert result["rejection_counts"] == {reason: 1}


def test_uniform_sampling_without_a_mask_retains_the_existing_contract():
    result = sample_region(**scene())
    assert result["sampling_policy"] == "uniform_grid_v1"
    assert result["samples"] == result["sampling_budget"] == 169
    assert result["candidates"]
    assert [c["center_rank"] for c in result["candidates"]] == sorted(
        c["center_rank"] for c in result["candidates"]
    )
