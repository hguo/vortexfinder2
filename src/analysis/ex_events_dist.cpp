#include "def.h"
#include "common/VortexTransition.h"
#include "common/VortexEvents.h"
#include "common/VortexLine.h"
#include <cstdio>
#include <cfloat>
#include <sstream>

static float Dist(const std::string& dataname, int frame, int lvid0, int lvid1)
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

  return MinimumDist(vortex_liness[lvid0], vortex_liness[lvid1]);
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
      int lvid[2];
      int j = 0;
      for (std::set<int>::const_iterator it = e.lhs.cbegin(); it != e.lhs.cend(); it ++) {
        lvid[j++] = *it;
      }

      int gvid[2] = {
        vt.lvid2gvid(e.frame, lvid[0]), 
        vt.lvid2gvid(e.frame, lvid[1])};

      float dist = Dist(dataname, e.frame, lvid[0], lvid[1]);

      fprintf(stderr, "frame=%d, gvid={%d, %d}, lvid={%d, %d}, dist=%f\n", e.frame, gvid[0], gvid[1], lvid[0], lvid[1], dist);
    }
  }

  return 0;
}
