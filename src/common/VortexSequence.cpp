#include "VortexSequence.h"
#include <set>
#include <cassert>

int VortexSequenceMap::SequenceID(int t, int lid) const
{
  std::tuple<int, int> key(t, lid);
  std::map<std::tuple<int,int>,int>::const_iterator it = _seqmap.find(key);
  if (it == _seqmap.end())
    return -1;
  else 
    return it->second;
}

int VortexSequenceMap::NewVortexSequence(int ts)
{
  VortexSequence vs; 
  vs.ts = ts;
  vs.tl = 1;
  push_back(vs);
  return size() - 1;
}

void VortexSequenceMap::Construct(const VortexTransition& vt, int ts, int tl)
{
  for (int i=ts; i<ts+tl-1; i++) {
    VortexTransitionMatrix tm = vt.at(i);
    assert(tm.Valid());

    const int n0 = tm.n0(), n1 = tm.n1(), 
              n = n0 + n1;
    
    if (i == ts) { // initial ids;
      for (int k=0; k<n0; k++) {
        int gid = NewVortexSequence(i);
        _seqmap[std::make_tuple(i, k)] = gid;
      }
    }
    
    std::set<int> unvisited; 
    for (int i=0; i<n; i++) 
      unvisited.insert(i);
 
    while (!unvisited.empty()) {
      std::set<int> lhs, rhs; 
      std::vector<int> Q;
      Q.push_back(*unvisited.begin());

      while (!Q.empty()) {
        int v = Q.back();
        Q.pop_back();
        unvisited.erase(v);
        if (v<n0) lhs.insert(v);
        else rhs.insert(v-n0);

        if (v<n0) { // left hand side
          for (int j=0; j<n1; j++) 
            if (tm(v, j)>0 && unvisited.find(j+n0) != unvisited.end())
              Q.push_back(j+n0);
        } else {
          for (int i=0; i<n0; i++) 
            if (tm(i, v-n0)>0 && unvisited.find(i) != unvisited.end())
              Q.push_back(i);
        }
      }

      if (lhs.size() == 1 && rhs.size() == 1) { // ordinary case
        int l = *lhs.begin(), r = *rhs.begin();
        int gid = _seqmap[std::make_tuple(i, l)];
        _seqmap[std::make_tuple(i+1, r)] = gid;
        at(gid).tl ++;
      } else { // events
        for (std::set<int>::iterator it=rhs.begin(); it!=rhs.end(); it++) {
          int r = *it;
          int gid = NewVortexSequence(i+1);
          _seqmap[std::make_tuple(i+1, r)] = gid;
        }
      }
     
#if 0
      else if (lhs.size() == 0 && rhs.size() == 1) {
        fprintf(stderr, "birth\n");
        int r = *rhs.begin();
        int gid = NewVortexSequence(i+1);
        seqmap[std::make_tuple(i+1, r)] = gid;
      } else if (lhs.size() == 1 && rhs.size() == 0) {
        fprintf(stderr, "death\n"); 
        // nothing todo
      } else if (lhs.size() == 1 && rhs.size() == 2) {
        fprintf(stderr, "split\n"); 
        int gid1 = NewVortexSequence(i+1), 
            gid2 = NewVortexSequence(i+1);
        int r1 = *rhs.begin(), 
            r2 = *(rhs.begin()+1);
        seqmap[std::make_tuple(i+1, r1)] = gid1;
        seqmap[std::make_tuple(i+1, r2)] = gid2;
      } else if (lhs.size() == 2 && rhs.size() == 1) {
        fprintf(stderr, "merge\n");
        int r = *rhs.begin(); 
        int gid = NewVortexSequence(i+1);
        seqmap[std::make_tuple(i+1, r)] = gid;
      } else if (lhs.size() > 1 && rhs.size() > 1) { 
        fprintf(stderr, "recombination\n");
        int gid1 = NewVortexSequence(i+1), 
            gid2 = NewVortexSequence(i+1);
        int r1 = *rhs.begin(), 
            r2 = *(rhs.begin()+1);
        seqmap[std::make_tuple(i+1, r1)] = gid1;
        seqmap[std::make_tuple(i+1, r2)] = gid2;
      }
#endif
#if 0
      int cnt=0;
      fprintf(stderr, "lhs={");
      for (std::set<int>::iterator it=lhs.begin(); it!=lhs.end(); it++) {
        if (cnt<lhs.size()-1) fprintf(stderr, "%d, ", *it);
        else fprintf(stderr, "%d}, ", *it);
        cnt ++;
      }
      cnt=0;
      fprintf(stderr, "rhs={");
      for (std::set<int>::iterator it=rhs.begin(); it!=rhs.end(); it++) {
        if (cnt<rhs.size()-1) fprintf(stderr, "%d, ", *it);
        else fprintf(stderr, "%d}\n", *it);
        cnt ++;
      }
#endif
    }
  }

  for (int i=0; i<size(); i++) 
    fprintf(stderr, "gid=%d, ts=%d, tl=%d\n",
        i, at(i).ts, at(i).tl);
}
