set (common_sources
  VortexObject.cpp
  FieldLine.cpp
)

set (common_protos
  DataInfo.proto
  VortexObject.proto
  FieldLine.proto
)

protobuf_generate_cpp (common_proto_sources common_proto_headers ${common_protos})

add_library (glcommon ${common_sources} ${common_proto_sources})
target_link_libraries (glcommon ${PROTOBUF_LIBRARY})
