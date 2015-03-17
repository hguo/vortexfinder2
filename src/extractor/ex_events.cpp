#include <iostream>
#include <vector>
#include <sstream>
#include <set>
#include "common/VortexTransition.h"
#include "common/VortexSequence.h"

int main(int argc, char **argv)
{
  if (argc<4) return 1; // ex_events dataname ts tl

  std::string dataname = argv[1]; 
  int ts = atoi(argv[2]), 
      tl = atoi(argv[3]);

  VortexTransition vt;
  vt.LoadFromFile(dataname, ts, tl);
  vt.SaveToDotFile("dot");

  VortexSequenceMap seq;
  seq.Construct(vt, ts, tl);

  // ExtractEvents(ts, tl);
  
  return EXIT_SUCCESS; 
}


#if 0
VortexTransition vt;

struct VortexSequence {
public:
  int ts, tl; // start and duration
};

std::vector<VortexSequence> seqs;
std::map<std::tuple<int, int>, int> seqmap; // <time, lid>, gid

int NewVortexSequence(int ts)
{
  VortexSequence vs; 
  vs.ts = ts;
  vs.tl = 1;
  seqs.push_back(vs);
  return seqs.size() - 1;
}

void LoadTransitionMatrix(const std::string &dataname, int t0, int t1)
{
  std::stringstream ss;
  ss << dataname << ".match." << t0 << "." << t1;

  VortexTransitionMatrix tm;
  if (tm.LoadFromFile(ss.str()))
    vt[t0] = tm;
}

void ExtractEvents(int ts, int tl)
{
  for (int i=ts; i<ts+tl-1; i++) {
    VortexTransitionMatrix tm = vt[i];
    // fprintf(stderr, "============%d, %d===========\n", tm.t0(), tm.t1());
    // fprintf(stderr, "n0=%d, n1=%d\n", tm.n0(), tm.n1());

    const int n0 = tm.n0(), n1 = tm.n1(), 
              n = n0 + n1;
    
    if (i == ts) { // initial ids;
      for (int k=0; k<n0; k++) {
        int gid = NewVortexSequence(i);
        seqmap[std::make_tuple(i, k)] = gid;
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
        int gid = seqmap[std::make_tuple(i, l)];
        seqmap[std::make_tuple(i+1, r)] = gid;
        seqs[gid].tl ++;
      } else { // events
        for (std::set<int>::iterator it=rhs.begin(); it!=rhs.end(); it++) {
          int r = *it;
          int gid = NewVortexSequence(i+1);
          seqmap[std::make_tuple(i+1, r)] = gid;
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

  for (int i=0; i<seqs.size(); i++) {
    fprintf(stderr, "vid=%d, ts=%d, tl=%d\n", i, seqs[i].ts, seqs[i].tl);
  }
}

int main(int argc, char **argv)
{
  if (argc<4) return 1; // ex_events dataname ts tl

  std::string dataname = argv[1]; 
  int ts = atoi(argv[2]), 
      tl = atoi(argv[3]);

  for (int i=ts; i<ts+tl-1; i++) 
    LoadTransitionMatrix(dataname, i, i+1);

  ExtractEvents(ts, tl);
  
  return EXIT_SUCCESS; 
}
#endif
