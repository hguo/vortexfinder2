#include "io/GLGPU3DDataset.h"

int main(int argc, char **argv)
{
  if (argc<2) return 1; 
  const std::string filename = argv[1]; // = "GL3D_Xfieldramp_inter_0437_cop.dat";

  GLGPU3DDataset *dataset = new GLGPU3DDataset; 
  dataset->OpenDataFile(filename);
  // dataset->WriteNetCDFFile(filename + ".nc");

  dataset->LoadTimeStep(0, 0);
  dataset->PrintInfo(0);
  for (int i=1; i<dataset->NTimeSteps(); i++) {
    dataset->LoadTimeStep(i, 1);
    dataset->PrintInfo(1);
    dataset->RotateTimeSteps();
  }

  delete dataset; 
  return 0; 
}
