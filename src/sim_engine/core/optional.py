"""Optional native runtimes, loaded on first use.

Habitat (0.3.1 or GS 0.3.3), vLLM/transformers and qwen-vl-utils each live in their own
environment, and ``serve-policy`` must set ``CUDA_VISIBLE_DEVICES`` before vLLM loads. They
are therefore the only modules not imported at the top of a file, and only through here.
"""

import importlib

RUNTIMES = frozenset({"habitat_sim", "magnum", "vllm", "transformers", "qwen_vl_utils"})


def optional_module(name):
    if name.split(".")[0] not in RUNTIMES:
        raise ValueError("not an optional runtime module: " + name)
    return importlib.import_module(name)
