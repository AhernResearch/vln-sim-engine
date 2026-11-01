import argparse
import asyncio
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stateful Habitat and asynchronous vLLM services")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, port in [("serve-sim", 18810), ("serve-policy", 18811)]:
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
    for name in ("prepare-navmeshes", "audit-navmesh"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "prepare-navmeshes":
            sub.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "rollout":
        from .client import rollout

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
    from .config import Settings

    settings = Settings.read(args.config)
    if args.command in ("prepare-navmeshes", "audit-navmesh"):
        from .client import atomic_json
        from .navmesh import audit_connectivity, prepare_navmeshes

        if args.command == "prepare-navmeshes":
            prepare_navmeshes(settings, args.dataset, args.source_root, args.output)
        else:
            result = audit_connectivity(settings, args.dataset)
            atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
            if not result["ready"]:
                raise SystemExit(2)
        return
    if args.command == "serve-policy":
        os.environ.update(
            CUDA_VISIBLE_DEVICES=str(args.gpu),
            HF_HUB_OFFLINE="1",
            MODELSCOPE_OFFLINE="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            TOKENIZERS_PARALLELISM="false",
        )
        from .policy_server import create_app
    else:
        from .sim_server import create_app
    import uvicorn

    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)
