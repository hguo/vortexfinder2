# README #

### Introduction ###

A visualization and analysis tool for condor2/GLGPU dataset

* Vortex detection and tracking
* Super-current line tracing

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
$ ./extractor_condor2
FATAL: input filename not given.
USAGE:
./extractor_condor2 -i <input_filename> [-o output_filename] [--nogauge] [-t=<t>] [-T=<T>] [-Kx=<Kx>] [-Bx=<Bx>] [-By=<By>] [-Bz=<Bz>]

  --verbose   verbose output
  --benchmark Enable benchmark
  --nogauge   Disable gauge transformation
  --Kx        Kx
  --B*        Magnetic field
  -t          Starting time step for the analysis
  -T          Number of time step for the analysis
```

``` shell
./extractor_glgpu
FATAL: input filename not given.
USAGE:
./extractor_glgpu -i <input_filename> [-o output_filename] [--nogauge]

  --verbose   verbose output
    --benchmark Enable benchmark
      --nogauge   Disable gauge transformation
```

To analyze the Condor2 example data (tslab.3.Bz0_02.Nt1000.lu.512.e), please run with all necessary arguments: 

``` shell
$ ./extractor_condor2 tslab.3.Bz0_02.Nt1000.lu.512.e --Bz 0.02 -t 600
```

To analyze the GLGPU example data (GL3D_Xfieldramp_inter_0437_cop.dat), please run:

``` shell
$ ./extractor_glgpu GL3D_Xfieldramp_inter_0437_cop.dat
```

By default, the output file is the input filename plus ".vortex" suffix, 
e.g. "tslab.3.Bz0_02.Nt1000.lu.512.e.vortex" and "GL3D_Xfieldramp_inter_0437_cop.dat.vortex". The output file could be 
visualized by the viewer if it is compiled: 

``` shell
$ ./viewer tslab.3.Bz0_02.Nt1000.lu.512.e.vortex
$ ./viewer GL3D_Xfieldramp_inter_0437_cop.dat.vortex
```

In the GUI, use left mouse button to rotate, and use wheel to zoom in/out. 

### TODOs ###

* Interface for in-situ analysis
* Vortex line tracking
* More features in the GUI, e.g. super current streamlines, inclusions, etc. 
* Plugins to production visualization tools, e.g. ParaView and VisIt

### Contact ###

* [Hanqi Guo](http://www.mcs.anl.gov/~hguo/), [hguo@anl.gov](mailto:hguo@anl.gov)
