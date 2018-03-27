#include "def.h"
#include <diy/serialization.hpp>
#include <ftk/ftkEvents.h>
#include <ftk/ftkTransition.h>
#include <sstream>
#include <cstdio>
  
ftkTransition vt;
std::vector<std::vector<float> > distMatrices;

void seqDist(int gvid0, int gvid1, int f0_, std::map<int, float>& dist) {
  const ftkSequence &s0 = vt.Sequences()[gvid0]; 
  const ftkSequence &s1 = vt.Sequences()[gvid1];
  int f0 = std::max(20000, std::max(s0.its, s1.its)), // limit: 20000
      f1 = std::min((int)distMatrices.size()-1, std::min(s0.its+s0.itl-1, s1.its+s1.itl-1));

  const int t0 = vt.Frames()[f0_];

  if (f0>f1) return;
  for (int f=f0; f<=f1; f++) {
    const int lvid0 = vt.gvid2lvid(f, gvid0);
    const int lvid1 = vt.gvid2lvid(f, gvid1);
    // fprintf(stderr, "f=%d, lvid0=%d, lvid1=%d\n", f, lvid0, lvid1);
    const int t = vt.Frames()[f];
    
    const int nv = sqrt(distMatrices[f].size());
    const float d = distMatrices[f][lvid0*nv + lvid1];
    dist[t-t0] = d; 
    
    // for (auto d : distMatrices[f]) fprintf(stderr, "%f ", d);
    // fprintf(stderr, "\n");
  }
}

int main(int argc, char **argv)
{
  if (argc < 2) return 1;

  rocksdb::DB* db;
  rocksdb::Options options;
  rocksdb::Status status = rocksdb::DB::OpenForReadOnly(options, argv[1], &db);

  vt.LoadFromDB(db);
  // vt.PrintSequence();

  for (int i=0; i<vt.Frames().size()-1; i++) {
    int f = vt.Frames()[i];
    std::stringstream ss;
    ss << "v." << f << "d." << f;
    std::string buf;
    std::vector<float> dist;
    
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), ss.str(), &buf);
    if (buf.size()>0) diy::unserialize(buf, dist);
    distMatrices.push_back(dist);
  }

  const std::vector<ftkEvent>& events = vt.Events();
  for (int i=0; i<events.size(); i++) {
    const ftkEvent& e = events[i];
    if (e.type == VORTEX_EVENT_RECOMBINATION) {
      std::vector<int> lhs, rhs;
      for (auto const &id : e.lhs) lhs.push_back(vt.lvid2gvid(e.if0, id));
      for (auto const &id : e.rhs) rhs.push_back(vt.lvid2gvid(e.if1, id));

      fprintf(stderr, "EVENT=%d, f0=%d, lhs={%d, %d}, f1=%d, rhs={%d, %d}\n", 
          i, e.if0, e.if1, lhs[0], lhs[1], rhs[0], rhs[1]);

      std::map<int, float> dist;
      seqDist(lhs[0], lhs[1], e.if0, dist);
      seqDist(rhs[0], rhs[1], e.if0, dist);

      for (std::map<int, float>::iterator it = dist.begin(); it != dist.end(); it ++) {
        int t = dist[0]>dist[1] ? it->first-1: it->first;
        if (t<-200 || t>200) continue;
        fprintf(stderr, "%d,%d,%f\n", i, t, it->second);
      }
    }
  }

  delete db;
  return 0;
}
