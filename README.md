# README #

## Introduction ##

A standalone tool and ParaView plugins to analyze and visualize vortices in Condor2/GLGPU data

We provide a set of tools: 

* Standalone command-line tools for extracting and tracking vortices
* Standalone OpenGL-based GUI for visualizing vortices
* ParaView plugins for analyzing and visualizing vorticies

## Build Guidelines ##

### Prerequisites ###

The following tools/libraries are mandatory for all tools:

* [CMake](http://www.cmake.org/) (mandatory, version >= 3.0.2)
* [Protocol Buffers](https://github.com/google/protobuf/) (mandatory, version >= 2.6.1), used for the serialization/unserialization of vortex objects

The following libraries are needed for analyzing Condor2 (unstructured mesh) data:

* [libMesh](http://libmesh.github.io/) (optional, preferable 0.9.4-rc1 built with PETSc), used for Condor2 data analysis
    * [PETSc](http://www.mcs.anl.gov/petsc/) (version >= 3.5.2), used for non-linear implicit system support in libMesh
    * [MPICH](http://www.mpich.org/) (version >= 3.1.2), PETSc dependency

The following libraries are needed for buiding the GUI in the standalone tools:

* [Qt4](http://www.qt.io/) (preferable 4.8.6), used for the GUI
* [GLEW](http://glew.sourceforge.net/), used for using OpenGL extensions

The following libraries are needed for buiding ParaView plugins: 

* [ParaView](http://www.paraview.org/)


### Build the command-line tools only ###

#### GLGPU data only ####

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build && cd build
$ cmake .. \
  -DPROTOBUF_ROOT=$PROTOBUF_INSTALL_DIR
$ make
```

#### GLGPU and Condor2 data ####

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build && cd build
$ cmake .. \
  -DWITH_LIBMESH=ON \
  -DPROTOBUF_ROOT=$PROTOBUF_INSTALL_DIR \
  -DMPI_HOME=$MPI_INSTALL_DIR \
  -DLIBMESH_DIR=$LIBMESH_INSTALL_DIR \
  -DPETSC_DIR=$PETSC_INSTALL_DIR 
$ make
```

### Build the standalone GUI (requires Qt4) ###

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build && cd build
$ cmake .. \
  -DWITH_QT=ON \
  -DPROTOBUF_ROOT=$PROTOBUF_INSTALL_DIR \
  -DQT_QMAKE_EXECUTABLE=$QT_INSTALL_DIR/bin/qmake 
$ make
```

### Build ParaView plugins only ###

Build ParaView first

``` shell
$ tar zxf ParaView-v4.3.1-source.tar.gz
$ cd ParaView-v4.3.1-source
$ export PARAVIEW_SOURCE_DIR=$PWD
$ mkdir build && cd build
$ export PARAVIEW_BUILD_DIR=$PWD
$ cmake ..
$ make
```

Build the plugins

``` shell
$ cd $VORTEX_FINDER2_SOURCE_DIR
$ mkdir build && cd build
$ cmake .. \
  -DWITH_PARAVIEW=ON \
  -DParaView_DIR=$PARAVIEW_BUILD_DIR \
  -DPROTOBUF_INCLUDE_DIR=$PARAVIEW_SOURCE_DIR/ThirdParty/protobuf/vtkprotobuf/src \
  -DPROTOBUF_LIBRARY=$PARAVIEW_BUILD_DIR/lib/libprotobuf.dylib \
  -DPROTOBUF_PROTOC_EXECUTABLE=$PARAVIEW_BUILD_DIR/bin/protoc
$ make
```

Then you get two binaries libBDATReader.dylib and libGLGPUVortexFilter.dylib in lib directory.
Install the plugins by loading them in the plugin manager in ParaView.


## Analyzing and visualizing GLGPU data with standalone tools ##

### Creating the file list ###

Create a file (e.g. GL3D_CrBx004_full_long) which contains a list of GLGPU data file. The frame ID is indexed by the line numbers. 

~~~
GL3D_CrBx004_full_long_0001_amph.dat
GL3D_CrBx004_full_long_0002_amph.dat
GL3D_CrBx004_full_long_0003_amph.dat
GL3D_CrBx004_full_long_0004_amph.dat
...
~~~


### Run the vortex extractor/tracker ###

``` shell
$ ../extractor_glgpu3D GL3D_CrBx004_full_long -t 0 -l 1000
```

The argument -t specifies the starting frame; -l specifies the number of frames. 
The tool then analyzes the data and store the results into a series of files. 
GL3D_CrBx004_full_long.pf.(i) are the punctured faces at frame i; 
GL3D_CrBx004_full_long.pe.(i).(i+1) are the intersected space-time edges for frame i and i+1; 
GL3D_CrBx004_full_long.vlines.(i) are the vortex lines at frame i; 
GL3D_CrBx004_full_long.match.(i).(i+1) are the correspondence of vortex IDs of frame i and i+1. 
This process may take a long time.


### Interactive 3D visualization ###

After the extraction and tracking, run the viewer for the 3D interactive visualization:

``` shell
$ ../viewer1 GL3D_CrBx004_full_long -t 0 -l 1000
```

In the viewer, use left mouse button to rotate, and wheel to zoom in/out. Press left/right key to show the previous/next frame.


## Analyzing and visualizing GLGPU data with ParaView plugins ##

- Install the two plugins, BDATReader and GLGPUVortexFilter. BDATReader can open GLGPU output files in both BDAT format and the legacy "CA02" format. 
- Open the .dat file with BDATReader
- Use GLGPUVortexFilter (Filters-->TDGL-->GLGPUVortexFilter in menu) to extract vortices
- Use Tube filter to make the vortex lines look better (optional)


## TODOs ##

* Interface for in-situ analysis
* More features in the GUI, e.g. super current streamlines, inclusions, etc. 


## References ##

H. Guo, C. L. Phillps, T. Peterka, D. Karpeyev, and A. Glatz. 
Extracting, Tracking and Visualizing Magnetic Flux Vortices in 3D Complex-Valued Superconductor Simulation Data. IEEE Trans. Vis. Comput. Graph. (SciVis '15), 21(12):-, 2015. (Accepted)

C. L. Phillps, T. Peterka, D. Karpeyev, and A. Glatz.
Detecting Vortices in Superconductors: Extracting One-Dimensional Topological Singularities from a Discretized Complex Scalar Field. Physics Review E, 023331(91):1-12, 2015.


## Contact ##

* [Hanqi Guo](http://www.mcs.anl.gov/~hguo/), [hguo@anl.gov](mailto:hguo@anl.gov)
