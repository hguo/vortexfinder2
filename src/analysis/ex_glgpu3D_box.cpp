#include <iostream>
#include <cstdio>
#include <vector>
#include <getopt.h>
#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

int main(int argc, char **argv)
{
  if (argc<5) return false;

  const std::string filename_in = argv[1];
  const int T0 = atoi(argv[2]);
  const int T1 = T0 + atoi(argv[3]);
  const int type = atoi(argv[4]); // 0:YZ, 1:ZX, 2:XY

  GLGPU3DDataset ds;
  ds.OpenDataFile(filename_in);
  ds.LoadTimeStep(T0, 0);
  ds.SetMeshType(GLGPU3D_MESH_HEX);
  ds.BuildMeshGraph();
  ds.PrintInfo();

  std::vector<FaceIdType> fids = ds.GetBoundaryFaceIds(type);

  VortexExtractor extractor;
  extractor.SetDataset(&ds);

  extractor.ExtractFaces(fids, 0);
  for (int t=T0+1; t<T1; t++) {
    ds.LoadTimeStep(t, 1);
    extractor.ExtractFaces(fids, 1);
    extractor.RotateTimeSteps();
    ds.RotateTimeSteps();
  }

  return EXIT_SUCCESS; 
}
