import sys
from glob import glob
from os import environ
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).parent.resolve()


def eigen_include_dirs():
    env_include = environ.get("EIGEN_INCLUDE")
    if env_include:
        return [env_include]

    candidates = [
        *glob("/opt/homebrew/Cellar/eigen/*/include/eigen3"),
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    return [str(Path(path)) for path in candidates if Path(path).exists()]


def openmp_flags():
    include_dirs = []
    compile_args = []
    link_args = []

    if sys.platform == "darwin":
        env_prefix = environ.get("LIBOMP_PREFIX")
        candidates = [Path(env_prefix)] if env_prefix else [
            Path(path) for path in (
                *glob("/opt/homebrew/opt/libomp"),
                *glob("/usr/local/opt/libomp"),
            )
        ]
        libomp = next((path for path in candidates if path.exists()), None)
        if libomp is not None:
            include_dirs.append(str(libomp / "include"))
            compile_args.extend(["-Xpreprocessor", "-fopenmp"])
            link_args.extend([
                f"-L{libomp / 'lib'}",
                "-lomp",
                f"-Wl,-rpath,{libomp / 'lib'}",
            ])
        else:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    return include_dirs, compile_args, link_args


openmp_include_dirs, openmp_compile_args, openmp_link_args = openmp_flags()


extensions = [
    Extension(
        "methods_cython.methods",
        sources=[
            "methods_cython/methods.pyx",
            "src/common/methods_common.cpp",
            "src/common/state_manager.cpp",
            "src/common/sparse_operations.cpp",
            "src/common/dropping.cpp",
            "src/common/result_operations.cpp",
            "src/common/methods_c_api.cpp",
            "src/global_spai/conjugate_gradient.cpp",
            "src/global_spai/minimal_residual.cpp",
            "src/global_spai/locally_minimal_residual.cpp",
            "src/inner_outer/minimal_residual.cpp",
        ],
        include_dirs=[
            "include",
            "src",
            "src/common",
            np.get_include(),
            *eigen_include_dirs(),
            *openmp_include_dirs,
        ],
        language="c++",
        extra_compile_args=["-std=c++20", "-O3", "-pthread", *openmp_compile_args],
        extra_link_args=["-pthread", *openmp_link_args],
    )
]


setup(
    name="matrix-methods",
    version="0.1.0",
    packages=["methods_cython"],
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
