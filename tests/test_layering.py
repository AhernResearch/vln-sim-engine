"""Package layering: every sim_engine import points to the same or a lower layer."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "sim_engine"
# A package may import only itself and the packages listed for it.
ALLOWED = {
    "core": set(),
    "navigation": {"core"},
    "planning": {"core", "navigation", "base"},
    "collection": {"core", "navigation", "planning", "base"},
    # Evaluation serving (sessions, policy, backends, scenes) at the package root.
    "base": {"core", "navigation"},
}
# The command line is the composition root that wires every service together.
COMPOSITION_ROOTS = {"cli.py"}


def imported_modules(path):
    package = path.relative_to(ROOT).with_suffix("").parts
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level]
                yield ".".join((*base, *(node.module.split(".") if node.module else ())))
            elif node.module and node.module.startswith("sim_engine."):
                yield node.module.removeprefix("sim_engine.")
        elif isinstance(node, ast.Import):
            yield from (a.name.removeprefix("sim_engine.") for a in node.names
                        if a.name.startswith("sim_engine."))


def layer(module):
    top = module.split(".")[0].removesuffix(".py")
    return top if top in ALLOWED and top != "base" else "base"


def test_packages_only_import_lower_layers():
    violations = []
    for package, allowed in ALLOWED.items():
        root = ROOT / package
        paths = (root.rglob("*.py") if package != "base" else
                 (p for p in ROOT.rglob("*.py") if layer(p.relative_to(ROOT).parts[0]) == "base"
                  and p.name not in COMPOSITION_ROOTS))
        for path in sorted(paths):
            for module in imported_modules(path):
                target = layer(module)
                if target != package and target not in allowed:
                    violations.append(f"{path.relative_to(ROOT)} -> {module}")
    assert violations == []
