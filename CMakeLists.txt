project (glvortex1)
cmake_minimum_required (VERSION 2.8)

option (WITH_TESTS "Build the tests" OFF)
option (WITH_CUDA "Build with CUDA" OFF)
option (WITH_LIBMESH "Build with libMesh" ON) 
option (WITH_PNETCDF "Build with parallel-netcdf" OFF)
option (WITH_QT "Build with Qt" ON)

set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake-modules)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package (MPI REQUIRED)
find_package (Protobuf REQUIRED)

set (LIBMESH_DIR "/usr" CACHE PATH "The installed path for libMesh")
set (LIBMESH_INCLUDE_DIR "${LIBMESH_DIR}/include")
set (LIBMESH_LIBRARY_DIR "${LIBMESH_DIR}/lib")
set (LIBMESH_LIBRARY "${LIBMESH_LIBRARY_DIR}/libmesh_opt.dylib")

set (PETSC_DIR "/usr" CACHE PATH "The installed path for PETSc")
set (PETSC_INCLUDE_DIR "${PETSC_DIR}/include")

include_directories (
  ${LIBMESH_INCLUDE_DIR}
  ${PETSC_INCLUDE_DIR}
  ${MPI_INCLUDE_PATH}
  ${PROTOBUF_INCLUDE_DIR}
)

if (WITH_QT)
  set (CMAKE_INCLUDE_CURRENT_DIR ON)
  find_package (OpenGL REQUIRED)
  find_package (Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
  include (${QT_USE_FILE})
  include_directories (
    ${QT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
  )
endif ()

subdirs (src)
