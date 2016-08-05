#include "def.h"
#include "common/VortexTransition.h"
#include "common/VortexEvents.h"
#include "common/VortexLine.h"
#include <cstdio>
#include <cfloat>
#include <sstream>

static float CrossingPoint(const std::string& dataname, int frame, int lvid0, int lvid1, float X[3])
{
  std::stringstream ss;
  ss << dataname << ".vlines." << frame;
  const std::string filename = ss.str();

  std::string info_bytes;
  std::vector<VortexLine> vortex_liness;
  if (!::LoadVortexLines(vortex_liness, info_bytes, filename)) {
    fprintf(stderr, "cannot load vlines %s\n", filename.c_str());
    return FLT_MAX;
  }

  return CrossingPoint(vortex_liness[lvid0], vortex_liness[lvid1], X);
  // return MinimumDist(vortex_liness[lvid0], vortex_liness[lvid1]);
}

int main(int argc, char **argv)
{
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <dataname> <ts> <tl>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string dataname = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]);

  VortexTransition vt;
  vt.LoadFromFile(dataname, ts, tl);
  vt.ConstructSequence();
  // vt.PrintSequence();

  const std::vector<VortexEvent>& events = vt.Events();
  for (int i=0; i<events.size(); i++) {
    const VortexEvent& e = events[i];
    if (e.type == VORTEX_EVENT_RECOMBINATION) {
      int llvid[2], rlvid[2];
      int j = 0;
      for (std::set<int>::const_iterator it = e.lhs.cbegin(); it != e.lhs.cend(); it ++) {
        llvid[j++] = *it;
      }

      j = 0;
      for (std::set<int>::const_iterator it = e.rhs.cbegin(); it != e.rhs.cend(); it ++) {
        rlvid[j++] = *it;
      }


      int lgvid[2] = {
        vt.lvid2gvid(e.frame, llvid[0]), 
        vt.lvid2gvid(e.frame, llvid[1])};
      int rgvid[2] = {
        vt.lvid2gvid(e.frame+1, rlvid[0]), 
        vt.lvid2gvid(e.frame+1, rlvid[1])};

      float X0[3], X1[3];
      CrossingPoint(dataname, e.frame, llvid[0], llvid[1], X0);
      CrossingPoint(dataname, e.frame+1, rlvid[0], rlvid[1], X1);

      // fprintf(stderr, "frame=%d, lhs={%d, %d}, rhs={%d, %d}, crossPt0={%f, %f, %f}, crossPt1={%f, %f, %f}\n", 
      //     e.frame, lgvid[0], lgvid[1], rgvid[0], rgvid[1], 
      //     X0[0], X0[1], X0[2], X1[0], X1[1], X1[2]);
      fprintf(stderr, "frame=%d, lhs={%d, %d}, rhs={%d, %d}, crossPt0={%f, %f, %f}\n",
          e.frame, lgvid[0], lgvid[1], rgvid[0], rgvid[1], 
          X0[0], X0[1], X0[2]);
    }
  }

  return 0;
}
