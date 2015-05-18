project (glvortex1)
cmake_minimum_required (VERSION 2.8)

option (WITH_TESTS "Build the tests" OFF)
option (WITH_CUDA "Build with CUDA" OFF)
option (WITH_LIBMESH "Build with libMesh" OFF) 
option (WITH_PNETCDF "Build with parallel-netcdf" OFF)
option (WITH_QT "Build with Qt" OFF)
option (WITH_CUDA "Build with CUDA" OFF)
option (WITH_VTK "Build with VTK" OFF)
option (WITH_PARAVIEW "Build paraview plugins" OFF)

set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake-modules)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# enable c++11
list(APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package (Protobuf REQUIRED)

include_directories (
  ${PROTOBUF_INCLUDE_DIR})

if (WITH_LIBMESH)
  find_package (MPI REQUIRED)
  
  set (LIBMESH_DIR "/usr" CACHE PATH "The installed path for libMesh")
  set (LIBMESH_INCLUDE_DIR "${LIBMESH_DIR}/include")
  set (LIBMESH_LIBRARY_DIR "${LIBMESH_DIR}/lib")

  if (APPLE)
    set (LIBMESH_LIBRARY "${LIBMESH_LIBRARY_DIR}/libmesh_opt.dylib")
    # suppose libmesh is built with netcdf, thus we can directly use netcdf
    set (NETCDF_LIBRARY "${LIBMESH_DIR}/lib/libnetcdf.dylib")
  else ()
    set (NETCDF_LIBRARY "${LIBMESH_DIR}/lib/libnetcdf.so")
    set (LIBMESH_LIBRARY "${LIBMESH_LIBRARY_DIR}/libmesh_opt.so")
  endif ()
  
  set (PETSC_DIR "/usr" CACHE PATH "The installed path for PETSc")
  set (PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
  
  include_directories (
    ${LIBMESH_INCLUDE_DIR}
    ${PETSC_INCLUDE_DIR}
    ${MPI_INCLUDE_PATH})
endif ()

if (WITH_QT)
  find_package (OpenGL REQUIRED)
  find_package (GLEW REQUIRED)
  find_package (GLUT REQUIRED)
  find_package (Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
endif ()

if (WITH_PARAVIEW)
  find_package (ParaView REQUIRED)
  include (${PARAVIEW_USE_FILE})
endif ()

if (WITH_VTK)
  find_package (VTK REQUIRED)
endif ()

if (WITH_CUDA)
  find_package (CUDA REQUIRED)
endif ()

subdirs (src)
