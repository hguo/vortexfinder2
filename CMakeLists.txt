project (vortexfinder2)
cmake_minimum_required (VERSION 2.8.12)

if (CMAKE_PROJECT_NAME STREQUAL "ParaView")

  # Build as a plugin within ParaView
  
  # Find and Include ParaView
  if(NOT ParaView_SOURCE_DIR)
    FIND_PACKAGE(ParaView REQUIRED)
    INCLUDE(${PARAVIEW_USE_FILE})
  else()
    include_directories(${VTK_INCLUDE_DIRS})
  endif()

  # use Boost, since C++11 is not supported by ParaView
  find_package (Boost REQUIRED COMPONENTS thread system)
  include_directories (${Boost_INCLUDE_DIR})

  # Set Protobuf flags to point to ParaView's Protobuf
  FIND_PACKAGE(Protobuf REQUIRED)
  set (PROTOBUF_INCLUDE_DIR ${ParaView_SOURCE_DIR}/ThirdParty/protobuf/vtkprotobuf/src)
  set (PROTOBUF_LIBRARY ${ParaView_BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}protobuf${CMAKE_SHARED_LIBRARY_SUFFIX})
  set (PROTOBUF_PROTOC_EXECUTABLE ${ParaView_BINARY_DIR}/bin/protoc)

  set (WITH_VTK ON)
  set (WITH_PARAVIEW ON)

else()

  # Standalone build

  option (WITH_TESTS "Build the tests" OFF)
  option (WITH_CUDA "Build with CUDA" OFF)
  option (WITH_LIBMESH "Build with libMesh" OFF) 
  option (WITH_PNETCDF "Build with parallel-netcdf" OFF)
  option (WITH_QT "Build with Qt" OFF)
  option (WITH_CUDA "Build with CUDA" OFF)
  option (WITH_VTK "Build with VTK" OFF)
  option (WITH_PARAVIEW "Build paraview plugins" OFF)
  option (WITH_CXX11 "Build with C++11" OFF)

  set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake-modules)
  set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
  set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
  set (CMAKE_INCLUDE_CURRENT_DIR ON)

  # Always use a full RPATH
  if(APPLE)
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

  # C++11/Boost
  if (WITH_CXX11)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif ()
  else ()
    find_package (Boost REQUIRED COMPONENTS thread system)
    include_directories (${Boost_INCLUDE_DIR})
  endif ()

  if (APPLE)
    set (CMAKE_MACOSX_RPATH ON)
    set (CMAKE_SKIP_BUILD_RPATH FALSE)
    set (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    list (FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
      "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if ("${isSystemDir}" STREQUAL "-1")
      set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif ()
  endif ()

  if (OPENMP_FOUND)
    list (APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
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
  endif ()

  if (WITH_CUDA)
    find_package (CUDA REQUIRED)
  endif ()

endif()

find_package (Protobuf REQUIRED)
include_directories (${PROTOBUF_INCLUDE_DIR})

# boost options
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_RUNTIME OFF)

# openmp
# find_package (OpenMP)
if (OPENMP_FOUND)
  list (APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif ()

# threads
find_package (Threads REQUIRED)
set (CMAKE_THREAD_PREFER_PTHREAD)

add_subdirectory (src)
