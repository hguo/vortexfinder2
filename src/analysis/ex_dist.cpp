#include "def.h"
#include "common/VortexTransition.h"
#include "common/VortexLine.h"
#include <cfloat>
#include <cstdio>

static double Dist(const std::string& dataname, int frame, int lvid0, int lvid1)
{
  std::stringstream ss;
  ss << dataname << ".vlines." << frame;
  const std::string filename = ss.str();

  std::string info_bytes;
  std::vector<VortexLine> vortex_liness;
  if (!::LoadVortexLines(vortex_liness, info_bytes, filename))
    return DBL_MAX;

  return MinimumDist(vortex_liness[lvid0], vortex_liness[lvid1]);
}

int main(int argc, char **argv)
{
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <dataname> <ts> <tl> <gvid0> <gvid1>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string dataname = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]);
  const int gvid0 = atoi(argv[4]), 
            gvid1 = atoi(argv[5]);

  VortexTransition vt;
  vt.LoadFromFile(dataname, ts, tl);
  vt.ConstructSequence();
  // vt.PrintSequence();

  for (int frame=ts; frame<ts+tl; frame++) {
    const int lvid0 = vt.gvid2lvid(frame, gvid0), 
              lvid1 = vt.gvid2lvid(frame, gvid1);
    if (lvid0 == -1 || lvid1 == -1) {
      // fprintf(stderr, "frame=%d, N/A\n", frame);
    }
    else {
      double dist = Dist(dataname, frame, lvid0, lvid1);
      fprintf(stderr, "frame=%d, dist=%f\n", frame, dist);
    }
  }

  return 0;
}
