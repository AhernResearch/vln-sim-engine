"""Required stable-ABI CPU geometry kernels for the navigation mainline."""

from setuptools import Extension, setup

setup(options={"bdist_wheel": {"py_limited_api": "cp310"}}, ext_modules=[Extension(
    "sim_engine.navigation._astar", ["src/sim_engine/navigation/_astar.cpp",
                                   "src/sim_engine/navigation/_queries.cpp",
                                   "src/sim_engine/navigation/_path.cpp"],
    depends=["src/sim_engine/navigation/_geometry.h"],
    define_macros=[("Py_LIMITED_API", "0x030A0000")],
    extra_compile_args=["-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math"],
    language="c++", py_limited_api=True,
)])
