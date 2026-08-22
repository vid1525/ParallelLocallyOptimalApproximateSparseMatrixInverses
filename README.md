This Project is an implementation part of the Master Thesis "Parallel Locally Optimal Approximate Sparse Matrix Inverses" with provided exprimental results of the iterative methods Global SPAI and Inner-Outer methods.

The purpose of this work is to implement parallelized methods for the preconditioning that improve iterative methods convergence, compare convergence rates of the algorithms with other similar works, test parallelization effectiveness for these methods, and define the domains where some of the tested method perform better than others. For the reference of the produced results sample datasets from the repository https://github.com/venkovic/matrix-market are used. Results has to be compared with results presented in https://arxiv.org/abs/2511.09753. This project contains implementation of the following methods:
 - Global SPAI Conjugate Gradient (CG)
 - Global SPAI Minimal Residual (MR)
 - Global SPAI Locally Minimal Residual (LOMR)
 - Inner-Outer MINRES
 - Inner-Outer MR
 - Inner-Outer LOMR

Commands to run the project:
1. Launch c++ test and build all dependencies / libraries for launching python scripts
```
make all
```
2. Clean up all dependencies / libraries
```
make clean
```
3. Build all python dependencies:
```
make python
```
4. Run only c++ test
```
make test-cpp
```

## Docker

Build the self-contained experiment image (including the C++ toolchain, Eigen,
OpenMP, Python dependencies, and compiled extension):
```
docker compose build
```
Run all configured scenarios:
```
docker compose run --rm experiments
```
The downloaded large datasets and results are kept in Docker named volumes, so
they persist across runs. To run one script instead, append its command, for
example:
```
docker compose run --rm experiments python scripts/methods_thread_speedup.py --datasets tri100eigs4k --tries 20
```

Project structure:
 - `main.py` is a script that launches all experimental scenarios to be presented in the Thesis (reads `config.json` file). **Building the project via make operation is required before launching experimental scenarios**. This script simplifies full testing of the project (single launch of `python3 main.py` instead of several launches of different scripts). **For running script for large matrix convergence test, launch the __scripts/fetch_large_spd_matrices.py__ to upload these datasets**.
 - `config.json` contains description of experimental scenarios
 - `scripts` contains different scripts for conducting experiments with the implemented methods
 - `src` contains implementation using C and C++ programming languages, for the effective matrix operations Eigen library is used https://libeigen.gitlab.io/eigen/docs-5.0/
 - `include` contains files to include for the usage of the implemented methods directly via C and C++ API
 - `methods_cython` contains Cython middleware API for data transfering between python API and C/C++ implementation (overall it copies structures provided in python code to the structures format used in C/C++ and does the same when result is received from the methods)
 - `tests` contains basic validation tests for C++


Results after the launch of the scripts will appear in the scripts/results* folder with respective folders descriptions.
