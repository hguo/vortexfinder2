#include "def.h"
#include "common/VortexTransition.h"
#include "common/VortexLine.h"
#include "io/GLGPU_IO_Helper.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cfloat>

static std::vector<float> timesteps;
static std::vector<std::string> filenames;
static std::vector<float> Bz;

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
    Bz.push_back(h.B[2]);
  }

  ifs.close();
  return true;
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

  LoadTimesteps(dataname);

  VortexTransition vt;
  vt.LoadFromFile(dataname, ts, tl);
  vt.ConstructSequence();
  // vt.PrintSequence();

  for (int frame=ts; frame<ts+tl; frame++) {
      fprintf(stderr, "%d, %f, %f, %d\n", frame, timesteps[frame], Bz[frame], vt.NVortices(frame));
  }

  return 0;
}
