#include <iostream>
#include <cstdio>
#include <vector>
#include <getopt.h>
#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

static std::string filename_in;
static int T0=0, T=1; // start and length of timesteps

int main(int argc, char **argv)
{
  if (argv<4) return false;

  filename_in = argv[1];
  T0 = atoi(argv[2]);
  T1 = atoi(argv[3]);

  GLGPU3DDataset ds;
  ds.OpenDataFile(filename_in);
  ds.LoadTimeStep(T0, 0);
  ds.SetMeshType(GLGPU3D_MESH_HEX);
  ds.BuildMeshGraph();
  ds.PrintInfo();
 
  VortexExtractor extractor;
  extractor.SetDataset(&ds);

  // extractor.ExtractFaces(0);
  for (int t=T0; t<T1; t++) {
    ds.LoadTimeStep(t, 1);
    // extractor.ExtractFaces(1);
    extractor.RotateTimeSteps();
    ds.RotateTimeSteps();
  }

  return EXIT_SUCCESS; 
}
