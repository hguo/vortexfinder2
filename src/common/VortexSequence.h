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
  int event_left, event_right;
  std::vector<int> links_right;
};

class VortexSequenceMap : public std::vector<VortexSequence>
{
public:
  void Construct(const VortexTransition& vt, int ts, int tl);
  int SequenceID(int t, int lid) const;

  int ts() const {return _ts;}
  int tl() const {return _tl;}

private:
  int NewVortexSequence(int ts);

private: 
  std::map<std::tuple<int, int>, int> _seqmap; // <time, lid>, gid
  int _ts, _tl;
};

#endif
