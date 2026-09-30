UNAME_S := $(shell uname -s)
VENV_PYTHON := $(firstword $(wildcard .venv/bin/python))
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),python3)

EIGEN_INCLUDE ?= $(firstword $(wildcard \
	/opt/homebrew/Cellar/eigen/*/include/eigen3 \
	/opt/homebrew/include/eigen3 \
	/usr/local/include/eigen3 \
	/usr/include/eigen3))

CXX ?= c++
CPPFLAGS ?= -Iinclude -Isrc -Isrc/common
CXXFLAGS ?= -O3 -std=c++20 -pthread
LDFLAGS ?=
LDLIBS ?=

ifneq ($(EIGEN_INCLUDE),)
CPPFLAGS += -I$(EIGEN_INCLUDE)
endif

ifeq ($(UNAME_S),Darwin)
LIBOMP_PREFIX ?= $(firstword $(wildcard /opt/homebrew/opt/libomp /usr/local/opt/libomp))

ifneq ($(LIBOMP_PREFIX),)
CPPFLAGS += -I$(LIBOMP_PREFIX)/include
CXXFLAGS += -Xpreprocessor -fopenmp
LDFLAGS += -L$(LIBOMP_PREFIX)/lib -Wl,-rpath,$(LIBOMP_PREFIX)/lib
LDLIBS += -lomp
else
CXXFLAGS += -fopenmp
LDLIBS += -fopenmp
endif
else
CXXFLAGS += -fopenmp
LDLIBS += -fopenmp
endif

METHODS_SOURCES = \
	src/common/methods_common.cpp \
	src/common/state_manager.cpp \
	src/common/sparse_operations.cpp \
	src/common/dropping.cpp \
	src/common/result_operations.cpp \
	src/common/methods_c_api.cpp \
	src/global_spai/conjugate_gradient.cpp \
	src/global_spai/minimal_residual.cpp \
	src/global_spai/locally_minimal_residual.cpp \
	src/inner_outer/minimal_residual.cpp

.PHONY: all check-openmp check-cpp test-cpp python clean

all: check-cpp python test-cpp

check-openmp:
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -dM -E -x c++ /dev/null | grep -q '^#define _OPENMP '
	@echo "OpenMP enabled for C++ compilation"

check-cpp: check-openmp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(METHODS_SOURCES)
	rm -f methods_common.o state_manager.o sparse_operations.o dropping.o result_operations.o methods_c_api.o conjugate_gradient.o minimal_residual.o locally_minimal_residual.o

test-cpp: check-openmp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test.cpp $(METHODS_SOURCES) $(LDFLAGS) $(LDLIBS) -o /tmp/test
	/tmp/test

python:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build
	rm -f methods_cython/global_spai.cpp methods_cython/methods.cpp methods_cython/*.so methods_cython/*.pyd
	rm -f methods_common.o state_manager.o sparse_operations.o dropping.o result_operations.o methods_c_api.o conjugate_gradient.o minimal_residual.o locally_minimal_residual.o
