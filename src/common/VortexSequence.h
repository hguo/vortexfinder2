#ifndef _VORTEX_SEQUENCE_H
#define _VORTEX_SEQUENCE_H

#include "common/VortexTransition.h"
#include <vector>
#include <map>

enum {
  VORTEX_EVENT_BIRTH = 0,
  VORTEX_EVENT_DEATH = 1,
  VORTEX_EVENT_MERGE = 2,
  VORTEX_EVENT_SPLIT = 3,
  VORTEX_EVENT_RECOMBINATION = 4
};

struct VortexSequence {
public:
  int ts, tl; // start and duration
  std::vector<int> lids; // local ids
};

#endif
