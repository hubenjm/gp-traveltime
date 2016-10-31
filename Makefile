.PHONY: default build clean
.SUFFIXES: .so .c .pyx

# Compilers and paths
PYTHON = python
CYTHON = cython
CC = gcc
CFLAGS = -shared -fPIC
CYTHONFLAGS = profile=False
PYTHONPATH=from sys import path; path.append("src");
PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())'}
MPI4PY_INCLUDE = ${shell ${PYTHON} -c 'import mpi4py; print(mpi4py.get_include())'}
NUMPY_INCLUDE = ${shell ${PYTHON} -c 'import numpy; print(numpy.get_include())'}

# Packages
src_directory = src

default: build

build: \
	src/fastsweeping.pyx src/fastsweeping.c src/fastsweeping.so \
	src/domain.pyx src/domain.c src/domain.so \

%.c : %.pyx
	${CYTHON} -X${CYTHONFLAGS} -I${MPI4PY_INCLUDE} -I${NUMPY_INCLUDE} $<

%.so: %.c
	${CC} ${CFLAGS} -I${PYTHON_INCLUDE} -I${NUMPY_INCLUDE}  -o $@ $<

clean:
	${RM} src/*.so src/*.c src/*.pyc
