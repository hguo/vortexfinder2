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

  GLGPU3DDataset ds;
  ds.OpenDataFile(filename_in);
  ds.LoadTimeStep(T0, 0);
  ds.SetMeshType(GLGPU3D_MESH_HEX);
  ds.BuildMeshGraph();
  ds.PrintInfo();

  std::vector<FaceIdType> fids_yz = ds.GetBoundaryFaceIds(0), 
                          fids_zx = ds.GetBoundaryFaceIds(1),
                          fids_xy = ds.GetBoundaryFaceIds(2);

  VortexExtractor extractor;
  extractor.SetDataset(&ds);

  int p0, n0, p1, n1, p2, n2;

  extractor.ExtractFaces(fids_yz, 0, p0, n0);
  extractor.ExtractFaces(fids_zx, 0, p1, n1);
  extractor.ExtractFaces(fids_xy, 0, p2, n2);
  fprintf(stderr, "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", T0, p0, n0, p1, n1, p2, n2);

  for (int t=T0+1; t<T1; t++) {
    ds.LoadTimeStep(t, 1);
    extractor.ExtractFaces(fids_yz, 1, p0, n0);
    extractor.ExtractFaces(fids_zx, 1, p1, n1);
    extractor.ExtractFaces(fids_xy, 1, p2, n2);
    fprintf(stderr, "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", t, p0, n0, p1, n1, p2, n2);

    extractor.RotateTimeSteps();
    ds.RotateTimeSteps();
  }

  return EXIT_SUCCESS; 
}
