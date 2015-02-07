#ifndef _EDGE_H
#define _EDGE_H

#include "def.h"
#include <vector>

struct Face;

/////
struct Edge {
  EdgeIdType id;
  NodeIdType node0, node1;
  // std::vector<NodeIdType> nodes;
  std::vector<const Face*> faces; // faces which contains this edge
  std::vector<int> face_chiralities; // 1 or -1
  std::vector<int> face_edge_id;
};

#endif
