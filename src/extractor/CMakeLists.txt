set (extractor_sources
  Extractor.cpp
)

add_library (glextractor STATIC ${extractor_sources})
target_link_libraries (glextractor PUBLIC glcommon glio ${CMAKE_THREAD_LIBS_INIT})
#install (TARGETS glextractor DESTINATION ${VTK_INSTALL_LIBRARY_DIR})

if (WITH_LIBMESH)
  add_executable (extractor_condor2 ex_condor2.cpp)
  target_link_libraries (extractor_condor2 PUBLIC glextractor)
endif ()

add_executable (extractor_glgpu3D ex_glgpu3D.cpp)
target_link_libraries (extractor_glgpu3D PUBLIC glextractor)

add_executable (extractor_glgpu2D ex_glgpu2D.cpp)
target_link_libraries (extractor_glgpu2D PUBLIC glextractor)

add_executable (extractor_presave ex_presave.cpp)
target_link_libraries (extractor_presave PUBLIC glextractor)
