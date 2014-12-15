#include "io/GLGPUDataset.h"

int main(int argc, char **argv)
{
  GLGPUDataset *dataset = new GLGPUDataset; 
  dataset->OpenDataFile("GL3D_Xfieldramp_inter_0437_cop.dat");
  dataset->ComputeSupercurrentField();

  delete dataset; 

  return 0; 
}
