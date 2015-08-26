set (common_sources
  MeshGraph.cpp
  MeshGraphRegular2D.cpp
  MeshGraphRegular3D.cpp
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
  VortexLine.proto
  VortexObject.proto
  Inclusions.proto
  FieldLine.proto
  Puncture.proto
)

protobuf_generate_cpp (common_proto_sources common_proto_headers ${common_protos})

add_library (glcommon STATIC ${common_sources} ${common_proto_sources})
target_include_directories(glcommon PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
if (WITH_CXX11)
  target_link_libraries (glcommon PUBLIC ${PROTOBUF_LIBRARY})
else ()
  target_link_libraries (glcommon PUBLIC ${PROTOBUF_LIBRARY} ${Boost_LIBRARIES})
endif ()
