#ifndef _VORTEX_SEQUENCE_H
#define _VORTEX_SEQUENCE_H

#include "common/VortexTransition.h"
#include <vector>
#include <map>

struct VortexSequence {
public:
  int ts, tl; // start and duration
  std::vector<int> lids; // local ids
};

#endif
