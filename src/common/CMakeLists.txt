set (common_sources
  MeshGraph.cpp
  MeshGraphRegular2D.cpp
  MeshGraphRegular3D.cpp
  VortexLine.cpp
  VortexEvent.cpp
  VortexTransitionMatrix.cpp
  VortexTransition.cpp
  VortexSequence.cpp
  Inclusions.cpp
  FieldLine.cpp
  Puncture.cpp
)

set (common_protos
  MeshGraph.proto
  DataInfo.proto
  VortexLine.proto
  VortexObject.proto
  VortexEvent.proto
  Inclusions.proto
  FieldLine.proto
  Puncture.proto
)

protobuf_generate_cpp (common_proto_sources common_proto_headers ${common_protos})

add_library (glcommon ${common_sources} ${common_proto_sources})
target_link_libraries (glcommon ${PROTOBUF_LIBRARY})
