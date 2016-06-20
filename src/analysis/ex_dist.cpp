#include "def.h"
#include "common/VortexTransition.h"
#include "common/VortexLine.h"
#include "io/GLGPU_IO_Helper.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstdio>

static std::vector<float> timesteps;
static std::vector<std::string> filenames;

static bool LoadTimesteps(const std::string& dataname)
{
  std::ifstream ifs; 
  ifs.open(dataname.c_str(), std::ifstream::in); 
  if (!ifs.is_open()) return false;
  
  filenames.clear();
  timesteps.clear();

  GLHeader h;
  char fname[1024];

  while (ifs.getline(fname, 1024)) {
    filenames.push_back(fname);

    bool succ = false;
    if (!succ) {
      succ = GLGPU_IO_Helper_ReadBDAT(
          fname, h, NULL, NULL, NULL, NULL, true);
    } 
    if (!succ) {
      succ = GLGPU_IO_Helper_ReadLegacy(
          fname, h, NULL, NULL, NULL, NULL, true);
    }
    if (!succ) {
      fprintf(stderr, "cannot open file: %s\n", fname);
      return false;
    }

    timesteps.push_back(h.time);
    // fprintf(stderr, "frame=%d, time=%f\n", filenames.size()-1, h.time);
  }

  ifs.close();
  return true;
}

static float Dist(const std::string& dataname, int frame, int lvid0, int lvid1)
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
  if (argc < 8) {
    fprintf(stderr, "Usage: %s <dataname> <ts> <tl> <gvid0> <gvid1> <gvid2> <gvid3>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string dataname = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]);
  const int gvid0 = atoi(argv[4]), 
            gvid1 = atoi(argv[5]), 
            gvid2 = atoi(argv[6]), 
            gvid3 = atoi(argv[7]);

  LoadTimesteps(dataname);

  VortexTransition vt;
  vt.LoadFromFile(dataname, ts, tl);
  vt.ConstructSequence();
  // vt.PrintSequence();

  for (int frame=ts; frame<ts+tl; frame++) {
    float dist;
    const int lvid0 = vt.gvid2lvid(frame, gvid0), 
              lvid1 = vt.gvid2lvid(frame, gvid1), 
              lvid2 = vt.gvid2lvid(frame, gvid2), 
              lvid3 = vt.gvid2lvid(frame, gvid3); 
    if (lvid0 != -1 && lvid1 != -1) 
      dist = Dist(dataname, frame, lvid0, lvid1);
    else if (lvid2 != -1 && lvid3 != -1)
      dist = Dist(dataname, frame, lvid2, lvid3);
    else 
      dist = -DBL_MAX;

    if (dist >= 0)
      fprintf(stderr, "%d, %f, %f\n", frame, timesteps[frame], dist);
  }

  return 0;
}
