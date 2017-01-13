#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

int main(int argc, char **argv)
{
  GLGPU3DDataset ds;
  ds.SetPrecomputeSupercurrent(true);
  
  ds.OpenDataFile(argv[1]);
  ds.LoadTimeStep(atoi(argv[2]));
  
  // ds.SetMeshType(GLGPU3D_MESH_HEX);
  ds.SetMeshType(GLGPU3D_MESH_TET);
  ds.BuildMeshGraph();
  
  // if (tet) ds.SetMeshType(GLGPU3D_MESH_TET);
  // else ds.SetMeshType(GLGPU3D_MESH_HEX);
  // ds.BuildMeshGraph();
 
  const std::string out = std::string(argv[1]) + "." + argv[2];
  ds.WriteRaw(out);
  ds.PrintInfo();
  
  VortexExtractor extractor;
  extractor.SetDataset(&ds);
  
  extractor.ExtractFaces(0);
  extractor.TraceOverSpace(0);
  extractor.SaveVortexLines(0);

#if 0
  VortexExtractor extractor;
  extractor.SetDataset(&ds);
  extractor.SetGaugeTransformation(!nogauge);

  if (nthreads != 0) 
    extractor.SetNumberOfThreads(nthreads);

  if (archive)
    extractor.SetArchive(true);

  if (gpu)
    extractor.SetGPU(true);
 
  extractor.ExtractFaces(0);
  extractor.TraceOverSpace(0);
  extractor.SaveVortexLines(0);
  for (int t=T0+span; t<T0+T; t+=span){
    ds.LoadTimeStep(t, 1);
    // ds.PrintInfo(1);
    extractor.ExtractFaces(1);
    extractor.TraceOverSpace(1);
    extractor.ExtractEdges();
    extractor.TraceOverTime();
    extractor.SaveVortexLines(1);
    extractor.RotateTimeSteps();
    ds.RotateTimeSteps();
  }
#endif
  return EXIT_SUCCESS; 
  return 0;
}
