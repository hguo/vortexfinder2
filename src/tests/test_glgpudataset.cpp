#include "io/GLGPUDataset.h"

int main(int argc, char **argv)
{
  const std::string filename = "GL3D_Xfieldramp_inter_0437_cop.dat";

  GLGPUDataset *dataset = new GLGPUDataset; 
  dataset->OpenDataFile(filename);
  dataset->ComputeSupercurrentField();
  dataset->WriteNetCDFFile(filename + ".nc");

  delete dataset; 

  return 0; 
}
