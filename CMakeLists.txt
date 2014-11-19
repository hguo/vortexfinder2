project (glvortex1)
cmake_minimum_required (VERSION 2.8)

set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_AUTOMOC ON)

# set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake-modules)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_LIBRARY_DIR}/lib)

find_package (OpenGL REQUIRED)
find_package (GLUT REQUIRED)
find_package (GLEW REQUIRED)
find_package (CUDA REQUIRED)
find_package (Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
find_package (MPI REQUIRED)
find_package (VTK REQUIRED)

set (LIBMESH_DIR "/usr" CACHE PATH "The installed path for libMesh")
set (LIBMESH_INCLUDE_DIR "${LIBMESH_DIR}/include")
set (LIBMESH_LIBRARY_DIR "${LIBMESH_DIR}/lib")
set (LIBMESH_LIBRARY "${LIBMESH_LIBRARY_DIR}/libmesh_opt.dylib")

include (${QT_USE_FILE})

include_directories (
  ${CMAKE_SOURCE_DIR}
  ${QT_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIR}
  ${LIBMESH_INCLUDE_DIR}
  ${MPI_INCLUDE_PATH}
)

link_directories (
  ${LIBMESH_LIBRARY_DIR}
)

subdirs (src)
