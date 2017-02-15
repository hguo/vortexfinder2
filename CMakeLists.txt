cmake_minimum_required (VERSION 3.1.3)
project (vortexfinder2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/diy/include)

if (CMAKE_PROJECT_NAME STREQUAL "ParaView")

  # Build as a plugin within ParaView
  
  # Find and Include ParaView
  if(NOT ParaView_SOURCE_DIR)
    FIND_PACKAGE(ParaView REQUIRED)
    INCLUDE(${PARAVIEW_USE_FILE})
  else()
    include_directories(${VTK_INCLUDE_DIRS})
  endif()

  set (WITH_PARAVIEW ON)

else()

  # Standalone build

  option (WITH_ARMA "Build with Armadillo" OFF)
  option (WITH_TESTS "Build the tests" OFF)
  option (WITH_CUDA "Build with CUDA" OFF)
  option (WITH_LIBMESH "Build with libMesh" OFF)
  option (WITH_NETCDF "Build with NetCDF" OFF)
  option (WITH_PNETCDF "Build with parallel-netcdf" OFF)
  option (WITH_QT "Build with Qt" OFF)
  option (WITH_VTK "Build with VTK" OFF)
  option (WITH_TBB "Build with TBB" OFF)
  option (WITH_ROCKSDB "Build with RocksDB" OFF)
  option (WITH_PARAVIEW "Build paraview plugins" OFF)
  option (WITH_FORTRAN "Enable fortran" OFF)
  option (WITH_MACOS_RPATH "Enable macOS rpath support" ON)

  set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
  set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
  set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
  set (CMAKE_INCLUDE_CURRENT_DIR ON)

  if (WITH_FORTRAN) 
    enable_language (Fortran)
    find_package (LAPACK REQUIRED)
  endif ()

  # Set up RPATH if requested.
  if(WITH_MACOS_RPATH)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
      "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif()
  endif()

  if (WITH_NETCDF)
    find_package (NetCDF REQUIRED)
    include_directories (${NETCDF_INCLUDES})
  endif ()

  if (WITH_TBB) 
    find_package (TBB REQUIRED)
    include_directories (${TBB_INCLUDE_DIR})
  endif ()

  if (WITH_ROCKSDB)
    find_package (RocksDB REQUIRED)
    include_directories (${RocksDB_INCLUDE_DIR})
  endif ()

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
    include_directories(${VTK_INCLUDE_DIRS})
  endif ()

  if (WITH_CUDA)
    find_package (CUDA REQUIRED)
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
  endif ()

endif()

# openmp
# find_package (OpenMP)
if (OPENMP_FOUND)
  list (APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif ()

# threads
find_package (Threads REQUIRED)
set (CMAKE_THREAD_PREFER_PTHREAD)

add_subdirectory (src)
