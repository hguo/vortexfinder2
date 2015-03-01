#ifndef _VORTEX_OBJECT_H
#define _VORTEX_OBJECT_H

#include <set>
#include <list>
#include "def.h"

struct VortexObject {
  int gid, id; // gid: global id; id: local (time) id
  int timestep;
  // std::map<FaceIdType, PuncturedFace> faces;
  std::set<FaceIdType> faces;
  std::vector<std::list<FaceIdType> > traces;

  VortexObject() : id(INT_MAX), gid(INT_MAX) {}
};

#endif
