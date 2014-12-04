set (vortex_sources
  vortex_object.cpp
)

set (vortex_protos
  vortex_object.proto
)

protobuf_generate_cpp (vortex_proto_sources vortex_proto_headers ${vortex_protos})

add_library (glvortex ${vortex_sources} ${vortex_proto_sources})
target_link_libraries (glvortex ${PROTOBUF_LIBRARY})
