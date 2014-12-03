# README #

### What is this repository for? ###

* Vortex detection for condor2/GLGPU written in C++

### Prerequisites ###

* CMake (version >= 3.0.2)
* PETSc (version >= 3.5.2)
* libMesh (preferable 0.9.4-rc1) built with PETSc
* Qt4 (optional for the viewer, preferable 4.8.6)

### Building and Running Examples ###

Build VortexFinder2 core and the GUI (requires Qt4): 

~~~~
mkdir build
cd build
cmake .. \
  -DMPI_HOME=~/local/mpich-3.1.2 \
  -DLIBMESH_DIR=~/local/libmesh-0.9.4-rc1 \
  -DPETSC_DIR=~/local/petsc-3.5.2 \
  -DQT_QMAKE_EXECUTABLE=~/local/Qt-4.8.6/bin/qmake
make
```

Build VortexFinder2 core only: 

~~~~
mkdir build
cd build
cmake .. \
  -DMPI_HOME=~/local/mpich-3.1.2 \
  -DLIBMESH_DIR=~/local/libmesh-0.9.4-rc1 \
  -DPETSC_DIR=~/local/petsc-3.5.2 \
  -DWITH_QT=OFF
make
```

### TODOs ###

* Add support for both tet/hex mesh
* Add support for GLGPU output formats

### Who do I talk to? ###

* Hanqi Guo, hguo@anl.gov
