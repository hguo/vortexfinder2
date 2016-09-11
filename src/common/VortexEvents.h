#ifndef _VORTEX_EVENTS_H
#define _VORTEX_EVENTS_H

#include <vector>
#include "common/Interval.h"

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
  int if0, if1;
  int type;
  std::set<int> lhs, rhs; // local ids.
  // std::vector<int> lhs_gids, rhs_gids;

  static const char* TypeToString(int e) {
    static const char* strs[7] = {
      "dummy", "birth", "death", "merge", "split", "recombination", "compound"};
    return strs[e];
  }
};

namespace diy {
  template <> struct Serialization<VortexEvent> {
    static void save(diy::BinaryBuffer& bb, const VortexEvent& m) {
      diy::save(bb, m.if0);
      diy::save(bb, m.if1);
      diy::save(bb, m.type);
      diy::save(bb, m.lhs);
      diy::save(bb, m.rhs);
    }

    static void load(diy::BinaryBuffer&bb, VortexEvent& m) {
      diy::load(bb, m.if0);
      diy::load(bb, m.if1);
      diy::load(bb, m.type);
      diy::load(bb, m.lhs);
      diy::load(bb, m.rhs);
    }
  };
}

#endif
