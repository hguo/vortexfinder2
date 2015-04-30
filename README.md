# README #

### Introduction ###

A visualization and analysis tool for condor2/GLGPU dataset

* Vortex detection and tracking
* Super-current line tracing

### Build Guidelines ###

#### Prerequisites ####

The following tools/libraries are needed for building the cores:

* [CMake](http://www.cmake.org/) (mandatory, version >= 3.0.2)
* [Protocol Buffers](https://github.com/google/protobuf/) (mandatory, version >= 2.6.1), used for the serialization/unserialization of vortex objects
* [libMesh](http://libmesh.github.io/) (optional, preferable 0.9.4-rc1 built with PETSc), used for Condor2 data analysis
    * [PETSc](http://www.mcs.anl.gov/petsc/) (version >= 3.5.2), used for non-linear implicit system support in libMesh
    * [MPICH](http://www.mpich.org/) (version >= 3.1.2), PETSc dependency

The following tools/libraries are needed for buiding the GUI:

* [Qt4](http://www.qt.io/) (preferable 4.8.6), used for the GUI
* [GLEW](http://glew.sourceforge.net/), used for using OpenGL extensions

The above tools could be installed with MacPorts or Homebrew.

#### Build the core only ####

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build
$ cd build
$ cmake .. \
  -DPROTOBUF_ROOT=~/local/protobuf-2.6.1 \
  -DMPI_HOME=~/local/mpich-3.1.2 \
  -DLIBMESH_DIR=~/local/libmesh-0.9.4-rc1 \
  -DPETSC_DIR=~/local/petsc-3.5.2 \
  -DWITH_QT=OFF
$ make
```

If you only need to analyze GLGPU data, set WITH_LIBMESH=OFF:

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build
$ cd build
$ cmake .. \
  -DPROTOBUF_ROOT=~/local/protobuf-2.6.1 \
  -DWITH_LIBMESH=OFF \
  -DWITH_QT=OFF
$ make
```

#### Build the core and the GUI (requires Qt4) ####

``` shell
$ mkdir build
$ cd build
$ cmake .. \
  -DPROTOBUF_ROOT=~/local/protobuf-2.6.1 \
  -DMPI_HOME=~/local/mpich-3.1.2 \
  -DLIBMESH_DIR=~/local/libmesh-0.9.4-rc1 \
  -DPETSC_DIR=~/local/petsc-3.5.2 \
  -DQT_QMAKE_EXECUTABLE=~/local/Qt-4.8.6/bin/qmake \
  -DPROTOBUF_ROOT=~/local/protobuf-2.6.1
$ make
```

### Analyzing GLGPU Data ###

#### Creating the file list ####

Create a file (e.g. GL3D_CrBx004_full_long) which contains a list of GLGPU data file. The frame ID is indexed by the line numbers. 

~~~
GL3D_CrBx004_full_long_0001_amph.dat
GL3D_CrBx004_full_long_0002_amph.dat
GL3D_CrBx004_full_long_0003_amph.dat
GL3D_CrBx004_full_long_0004_amph.dat
GL3D_CrBx004_full_long_0005_amph.dat
GL3D_CrBx004_full_long_0006_amph.dat
GL3D_CrBx004_full_long_0007_amph.dat
GL3D_CrBx004_full_long_0008_amph.dat
GL3D_CrBx004_full_long_0009_amph.dat
GL3D_CrBx004_full_long_0010_amph.dat
...
~~~

#### Running the vortex extractor/tracker ####

``` shell
$ ../extractor_glgpu3D GL3D_CrBx004_full_long -t 0 -l 1000
```

The argument -t specifies the starting frame; -l specifies the number of frames. Then the program generates a series of files. GL3D_CrBx004_full_long.pf.(i) are the punctured faces at frame i; GL3D_CrBx004_full_long.pe.(i).(i+1) are the intersected space-time edges for frame i and i+1; GL3D_CrBx004_full_long.vlines.(i) are the vortex lines at frame i; GL3D_CrBx004_full_long.match.(i).(i+1) are the correspondence of vortex IDs of frame i and i+1. This process may take a long time.

#### Interactive 3D visualization ####

After the extraction and tracking, run the viewer for the 3D interactive visualization:

``` shell
$ ../viewer1 GL3D_CrBx004_full_long -t 0 -l 1000
```

In the viewer, use left mouse button to rotate, and wheel to zoom in/out. Press left/right key to show the previous/next frame.

### TODOs ###

* Interface for in-situ analysis
* More features in the GUI, e.g. super current streamlines, inclusions, etc. 
* Plugins to production visualization tools, e.g. ParaView and VisIt

### Contact ###

* [Hanqi Guo](http://www.mcs.anl.gov/~hguo/), [hguo@anl.gov](mailto:hguo@anl.gov)
