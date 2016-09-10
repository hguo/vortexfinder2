set (common_sources
  MeshGraph.cpp
  MeshGraphRegular2D.cpp
  MeshGraphRegular3D.cpp
  MeshGraphRegular3DTets.cpp
  VortexLine.cpp
  VortexTransitionMatrix.cpp
  VortexTransition.cpp
  Inclusions.cpp
  FieldLine.cpp
  Puncture.cpp
  zcolor.cpp
  random_color.cpp
  graph_color.cpp
)

set (common_protos
  MeshGraph.proto
  DataInfo.proto
  Inclusions.proto
  FieldLine.proto
  Puncture.proto
)

add_library (glcommon STATIC ${common_sources} ${common_proto_sources})
target_include_directories(glcommon PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

set (deps ${CMAKE_THREAD_LIBS_INIT})
if (WITH_VTK)
  list (APPEND deps ${VTK_LIBRARIES})
endif ()
if (WITH_ROCKSDB)
  list (APPEND deps ${RocksDB_LIBRARY})
endif ()

target_link_libraries (glcommon PUBLIC ${deps})
