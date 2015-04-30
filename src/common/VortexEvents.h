#ifndef _VORTEX_EVENTS_H
#define _VORTEX_EVENTS_H

#include <vector>

enum {
  VORTEX_EVENT_DUMMY = 0,
  VORTEX_EVENT_BIRTH = 1,
  VORTEX_EVENT_DEATH = 2,
  VORTEX_EVENT_MERGE = 3,
  VORTEX_EVENT_SPLIT = 4,
  VORTEX_EVENT_RECOMBINATION = 5, 
  VORTEX_EVENT_COMPOUND = 6
};

struct VortexEvent {
  int t, event;
  std::set<int> lhs, rhs; // local ids.
  // std::vector<int> lhs_gids, rhs_gids;
};

#endif
