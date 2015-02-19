#include "io/GLGPU2DDataset.h"

int main(int argc, char **argv)
{
  if (argc<2) return 1; 
  const std::string filename = argv[1]; // = "GL3D_Xfieldramp_inter_0437_cop.dat";

  GLGPU2DDataset *dataset = new GLGPU2DDataset; 
  dataset->OpenDataFile(filename);
  // dataset->WriteNetCDFFile(filename + ".nc");

  dataset->LoadTimeStep(0, 0);
  for (int i=1; i<dataset->NTimeSteps(); i++) {
    dataset->LoadTimeStep(i, 1);
  }

  delete dataset; 
  return 0; 
}
