#include "io/GLGPUDataset.h"

int main(int argc, char **argv)
{
  if (argc<2) return 1; 
  const std::string filename = argv[1]; // = "GL3D_Xfieldramp_inter_0437_cop.dat";

  GLGPUDataset *dataset = new GLGPUDataset; 
  dataset->OpenDataFile(filename);
  dataset->WriteNetCDFFile(filename + ".nc");

  delete dataset; 

  return 0; 
}
