import argparse
import asyncio
import json
import os
from pathlib import Path

import uvicorn

from . import policy_server, sim_server
from .acceptance.depth import validate_depth
from .acceptance.probe import probe_backend
from .client import rollout
from .collection import server as collection_server
from .collection.start_pool import build_pool
from .config import Settings
from .core.io import file_sha256, replace_json
from .navmesh import audit_connectivity, prepare_navmeshes
from .planning import service as planning_service
from .planning.assets import export_assets
from .planning.contracts import PlannerSettings

# Spawned simulation, planning and collection workers inherit these before importing NumPy,
# so small geometry operations do not start a BLAS thread pool per CPU core.
SINGLE_THREAD_NUMERICS = dict(
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stateful Habitat and asynchronous vLLM services")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, port in [("serve-sim", 18810), ("serve-policy", 18811), ("serve-collection", 18812), ("serve-planner", 18814)]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--host", default="127.0.0.1")
        sub.add_argument("--port", type=int, default=port)
        if name == "serve-policy":
            sub.add_argument("--gpu", type=int, required=True)
    run = commands.add_parser("rollout")
    run.add_argument("--sim-url", required=True)
    run.add_argument("--policy-url", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--episodes", nargs="+", required=True)
    run.add_argument("--camera", default="trained_ce30")
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-plans", type=int)
    run.add_argument("--output", type=Path, required=True)
    for name in ("prepare-navmeshes", "audit-navmesh", "export-planner-assets"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name != "export-planner-assets":
            sub.add_argument("--dataset", required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "prepare-navmeshes":
            sub.add_argument("--source-root", type=Path, required=True)
    prepare = commands.add_parser("prepare-scan")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--lod", type=int, default=3)
    bake = commands.add_parser("bake-scan")
    bake.add_argument("--manifest", type=Path, required=True)
    bake.add_argument("--agent-radius-m", type=float, default=0.18)
    bake.add_argument("--agent-height-m", type=float, default=1.5)
    depth = commands.add_parser("validate-depth")
    depth.add_argument("--output", type=Path, required=True)
    depth.add_argument("--gpu", type=int, required=True)
    probe = commands.add_parser("probe-backend")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    pool = commands.add_parser("build-start-pool")
    pool.add_argument("--config", type=Path, required=True)
    pool.add_argument("--dataset", required=True)
    pool.add_argument("--camera", required=True)
    pool.add_argument("--owner", required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--successes", type=int, default=100000)
    pool.add_argument("--minimum-per-scene", type=int, default=500)
    pool.add_argument("--attempt-factor", type=float, default=3.0)
    pool.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args(argv)
    if args.command == "prepare-scan":
        print(prepare(args.archive, args.output, args.lod))
        return
    if args.command == "bake-scan":
        print(json.dumps(bake(args.manifest, args.agent_radius_m, args.agent_height_m), indent=2))
        return
    if args.command == "validate-depth":
        report = validate_depth(args.output, args.gpu)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(2)
        return
    if args.command == "rollout":
        result = asyncio.run(
            rollout(
                args.sim_url,
                args.policy_url,
                args.dataset,
                args.episodes,
                args.camera,
                args.output,
                args.concurrency,
                args.seed,
                args.max_plans,
            )
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "serve-planner":
        os.environ.update(SINGLE_THREAD_NUMERICS)
        app = planning_service.create_app(PlannerSettings.read(args.config))
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
        return
    settings = Settings.read(args.config)
    if args.command == "probe-backend":
        print(json.dumps(probe_backend(settings, args.output), indent=2))
        return
    if args.command == "build-start-pool":
        print(json.dumps(build_pool(
            settings, file_sha256(args.config), args.dataset, args.camera, args.output, args.owner,
            successes=args.successes, minimum_per_scene=args.minimum_per_scene,
            attempt_factor=args.attempt_factor, seed=args.seed), indent=2))
        return
    if args.command == "export-planner-assets":
        print(json.dumps(export_assets(settings, args.output), indent=2))
        return
    if args.command in ("prepare-navmeshes", "audit-navmesh"):
        if args.command == "prepare-navmeshes":
            prepare_navmeshes(settings, args.dataset, args.source_root, args.output)
        else:
            result = audit_connectivity(settings, args.dataset)
            replace_json(args.output, result)
            print(json.dumps(result, indent=2))
            if not result["ready"]:
                raise SystemExit(2)
        return
    if args.command == "serve-policy":
        # vLLM is imported lazily inside the policy engine, after the device mask is set.
        os.environ.update(
            CUDA_VISIBLE_DEVICES=str(args.gpu),
            HF_HUB_OFFLINE="1",
            MODELSCOPE_OFFLINE="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            TOKENIZERS_PARALLELISM="false",
        )
        app = policy_server.create_app(settings)
    elif args.command == "serve-collection":
        os.environ.update(SINGLE_THREAD_NUMERICS)
        app = collection_server.create_app(settings)
    else:
        app = sim_server.create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
