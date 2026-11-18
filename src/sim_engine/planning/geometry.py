"""NavMesh-only geometry context. Never constructs a Simulator or requires a GLB."""

import math

from sim_engine.collection.sampling import ReferenceCache, select_pose
from sim_engine.collection.storage import RejectedCandidate
from sim_engine.core.io import create_json
from sim_engine.planning.context import checkpoint


class NavMeshContext:
    def __init__(self, navmesh, scene_id, camera, config, profile=None):
        from dataclasses import replace

        import habitat_sim as hs
        import numpy as np

        from sim_engine.collection.clearance import ClearancePlanner, ClearanceProfile
        from sim_engine.collection.tracking import expert_profile

        self.hs, self.np = hs, np
        self.scene_id, self.scene = scene_id, scene_id
        self.camera, self.config, self.tasks = camera, config, {}
        self.pf = hs.PathFinder()
        if not self.pf.load_nav_mesh(str(navmesh)):
            raise ValueError("cannot load pinned NavMesh")
        profile = profile or ClearanceProfile()
        optimize = config.get("planning_optimizations", False)
        search_backend = config.get("search_backend", "python")
        self.queries = None
        if config.get("query_optimizations", False):
            from .queries import NavMeshQueries

            self.queries = NavMeshQueries(self.pf)
        native_path = config.get("native_path_optimizations", False)
        self.planner = ClearancePlanner(self.pf, self.shortest, profile, optimize=optimize,
                                        search_backend=search_backend, query_cache=self.queries,
                                        native_path=native_path)
        minimum = config["minimum_planning_clearance_m"]
        self.planners = [self.planner]
        if minimum < profile.clearance_m:
            self.planners.append(ClearancePlanner(
                self.pf, self.shortest,
                replace(profile, clearance_m=minimum, grid_m=min(profile.grid_m, 0.05)),
                optimize=optimize, search_backend=search_backend, query_cache=self.queries,
                native_path=native_path,
            ))
        self.guard = ClearancePlanner(
            self.pf, self.shortest, replace(profile, clearance_m=minimum, edge_sample_m=0.0125),
            optimize=optimize, search_backend=search_backend, query_cache=self.queries,
            native_path=native_path,
        )
        self.expert_profile = expert_profile(config)
        self.references = ReferenceCache()

    def clear_task(self):
        # Share queries between a task's candidate searches, never across tasks.
        for planner in [*self.planners, self.guard]:
            planner.clear_queries()
        if self.queries is not None:
            self.queries.clear()
        self.references.values.clear()
        self.tasks.clear()

    def shortest(self, a, b):
        checkpoint(query=1)
        path = self.hs.ShortestPath()
        path.requested_start, path.requested_end = a, b
        if not self.pf.find_path(path):
            raise RejectedCandidate("anchors disconnected")
        return [self.np.asarray(p) for p in path.points]

    def reference(self, task, index):
        """Source samples guide the route; only the task goal is fixed."""
        from sim_engine.collection.guidance import plan_with_preferences

        if "reference_guides" in task:
            clean = task["reference_guides"][:-1]
        else:
            rows = task["rows"]
            guides = [rows[index]["world_position"]] + [
                r["world_position"] for r in rows[index + 1 :: 4]
            ]
            clean = [guides[0]]
            for p in guides[1:]:
                if math.dist(p, clean[-1]) > 0.01:
                    clean.append(p)
        try:
            prepared = plan_with_preferences(
                self.planners,
                self.guard,
                clean,
                task["goal_position"],
                self.config["guide_adjustment_radius_m"],
                self.config["endpoint_adjustment_radius_m"],
            )
        except ValueError as error:
            return {
                "status": "infeasible",
                "reason": str(error),
                "required_clearance_m": self.planner.profile.clearance_m,
            }
        return prepared

    def prepare(self, candidate, folder):
        from sim_engine.collection.tracking import preview

        task = self.tasks[candidate["task_id"]]
        if candidate["scene"] != self.scene or task["scene"] != self.scene:
            raise ValueError("scene affinity violation")
        if "reference_guides" in task:
            reference, reused = self.references.get(
                (task["id"], "sparse"), lambda: self.reference(task, None)
            )
            index = None
        else:
            index = candidate["source_frame"]
            reference, reused = self.references.get(
                (task["id"], index), lambda: self.reference(task, index)
            )
        create_json(
            folder / "reference-check.json",
            dict(reference, cached=reused, task_id=task["id"], source_frame=index),
        )
        if reference["status"] != "feasible":
            raise RejectedCandidate("reference_path_infeasible: " + reference["reason"])
        planner = self.planners[reference["planner_index"]]
        from dataclasses import asdict

        create_json(folder / "expert-profile.json", asdict(self.expert_profile))
        if "reference_guides" in task:
            from sim_engine.collection.reference import reference_state

            selected = reference_state(
                reference["path"], candidate["reference_fraction"], task["source_start_yaw"]
            )
            index, center, source_yaw = (
                selected["path_index"],
                self.np.asarray(selected["world_position"]),
                selected["yaw"],
            )
            remaining_path = reference["path"][index:]
            create_json(folder / "reference-state.json", selected)
        else:
            center = self.np.asarray(task["rows"][index]["world_position"])
            source_yaw = task["rows"][index]["base_pose_se2"][2]
            remaining_path = reference["path"]

        def evaluate(draw):
            point = center + draw["offset"]
            snapped = self.np.asarray(self.pf.snap_point(point))
            if (
                not self.np.isfinite(snapped).all()
                or self.np.linalg.norm(snapped - point) > 0.03
                or self.np.linalg.norm(snapped - center) > self.config["start_radius_m"]
            ):
                raise RejectedCandidate("start snap outside limits")
            if not self.pf.is_navigable(snapped):
                raise RejectedCandidate("start not navigable")
            if planner.clearance(snapped) + 1e-6 < planner.profile.clearance_m:
                raise RejectedCandidate("start clearance insufficient")
            if not self.guard.edge_clear(center, snapped):
                raise RejectedCandidate("unsafe local perturbation")
            yaw = source_yaw + draw["yaw_delta"]
            ep = {
                "episode_id": candidate["id"],
                "_dataset": "train_pilot",
                "scene_id": self.scene_id,
                "start_position": snapped.tolist(),
                "start_rotation": [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)],
                "goals": [{"position": reference["fixed_goal"], "radius": 0.3}],
            }
            try:
                # Connect to the adjusted safe guide, not the potentially unsafe old center.
                connector = planner.prepare([snapped.tolist(), remaining_path[0]])["world_points"]
                from sim_engine.collection.guidance import dense_surface_path

                path, _ = dense_surface_path(planner, connector + remaining_path[1:])
                forecast = preview(
                    self.pf, ep, path, self.expert_profile, max_steps=self.config["max_steps"]
                )
            except ValueError as error:
                raise RejectedCandidate(str(error)) from error
            if forecast["termination"] != "expert_stop" or forecast["collision_steps"]:
                raise RejectedCandidate("preview failed")
            query = self.shortest(
                forecast["rows"][-1]["world_position"], reference["requested_goal_surface"]
            )
            forecast["final_requested_goal_distance_m"] = sum(
                math.dist(a, b) for a, b in zip(query, query[1:])
            )
            if forecast["final_requested_goal_distance_m"] > self.config["goal_tolerance_m"]:
                raise RejectedCandidate("preview stops outside requested goal region")
            if not all(
                self.guard.edge_clear(a["world_position"], b["world_position"])
                for a, b in zip(forecast["rows"], forecast["rows"][1:])
            ):
                raise RejectedCandidate("preview lacks swept clearance")
            return {
                "episode": ep,
                "path": path,
                "yaw": yaw,
                "forecast": forecast,
                "requested_goal_position": reference["requested_goal_surface"],
                "goal_adjustment_m": reference["goal_adjustment_m"],
                "endpoint_review_required": reference["endpoint_review_required"],
            }

        value = select_pose(candidate, self.config, evaluate, folder)
        create_json(folder / "preflight.json", value.pop("forecast"))
        create_json(folder / "private-geometry.json", value)
        return value

