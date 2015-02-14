set (common_sources
  MeshGraph.cpp
  VortexLine.cpp
  FieldLine.cpp
)

set (common_protos
  MeshGraph.proto
  DataInfo.proto
  VortexLine.proto
  FieldLine.proto
  Puncture.proto
)

protobuf_generate_cpp (common_proto_sources common_proto_headers ${common_protos})

add_library (glcommon ${common_sources} ${common_proto_sources})
target_link_libraries (glcommon ${PROTOBUF_LIBRARY})
