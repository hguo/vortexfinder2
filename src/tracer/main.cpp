#include <iostream>
#include "io/GLGPUDataset.h"
#include "GLGPUTracer.h"

using namespace std; 

int main(int argc, char **argv)
{
  GLGPUDataset ds;
  ds.OpenDataFile("GL3D_Xfieldramp_inter_0437_cop.dat");

  GLGPUFieldLineTracer tracer;
  tracer.SetDataset(&ds);
  tracer.Trace();

  return 0;
}
