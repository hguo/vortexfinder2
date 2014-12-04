# README #

### Introduction ###

* Vortex detection for condor2/GLGPU written in C++

### Build Guidelines ###

#### Prerequisites ####

* [CMake](http://www.cmake.org/) (version >= 3.0.2)
* [MPICH](http://www.mpich.org/) (version >= 3.1.2), PETSc dependency
* [PETSc](http://www.mcs.anl.gov/petsc/) (version >= 3.5.2), for building non-linear implicit systems in libMesh
* [libMesh](http://libmesh.github.io/) (preferable 0.9.4-rc1) built with PETSc
* [Protocol Buffers](https://github.com/google/protobuf/) (version >= 2.6.1), for the serialization/unserialization of vortex objects
* [Qt4](http://qt-project.org/) (optional for the viewer, preferable 4.8.6)

#### Build the core only ####

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build
$ cd build
$ cmake .. \
  -DMPI_HOME=~/local/mpich-3.1.2 \
  -DLIBMESH_DIR=~/local/libmesh-0.9.4-rc1 \
  -DPETSC_DIR=~/local/petsc-3.5.2 \
  -DWITH_QT=OFF
$ make
```

#### Build the core and the viewer (requires Qt4) ####

``` shell
$ mkdir build
$ cd build
$ cmake .. \
  -DMPI_HOME=~/local/mpich-3.1.2 \
  -DLIBMESH_DIR=~/local/libmesh-0.9.4-rc1 \
  -DPETSC_DIR=~/local/petsc-3.5.2 \
  -DQT_QMAKE_EXECUTABLE=~/local/Qt-4.8.6/bin/qmake \
  -DPROTOBUF_ROOT=~/local/protobuf-2.6.1
$ make
```

### Running Examples ###

Change the working directory:

``` shell
cd build/bin
```

To show help information, run the command without arguments: 

``` shell
$ ./extractor
FATAL: input filename not given.
USAGE:
./extractor -i <input_filename> [-o output_filename] [-gauge] [-t=<t>] [-T=<T>] [-Kx=<Kx>] [-Bx=<Bx>] [-By=<By>] [-Bz=<Bz>]

  --verbose   verbose output
  --benchmark Enable benchmark
  --gauge     Enable gauge transformation
  --Kx        Kx
  --B*        Magnetic field
  -t          Starting time step for the analysis
  -T          Number of time step for the analysis
```

To analyze the example data (tslab.3.Bz0_02.Nt1000.lu.512.e), please add all necessary arguments: 

``` shell
$ ./extractor tslab.3.Bz0_02.Nt1000.lu.512.e --Bz 0.02 -t 600
```

By default, the output file is the input filename plus ".vortex" suffix, 
e.g. "tslab.3.Bz0_02.Nt1000.lu.512.e.vortex". The output file could be 
visualized by the viewer if it is compiled: 

``` shell
$ ./viewer tslab.3.Bz0_02.Nt1000.lu.512.e.vortex
```

In the GUI, use left mouse button to rotate, and use wheel to zoom in/out. 

### TODOs ###

* Add support for both tet/hex mesh
* Add support for GLGPU output formats (conversion to libMesh format)
* Interface for in-situ analysis
* Vortex line tracking
* More features in the GUI, e.g. super current streamlines, inclusions, etc. 
* Plugins to production visualization tools, e.g. ParaView and VisIt

### Contact ###

* [Hanqi Guo](http://www.mcs.anl.gov/~hguo/), [hguo@anl.gov](mailto:hguo@anl.gov)
