#include <iostream>
#include <cstdio>
#include "io/GLGPU3DDataset.h"
#include "tracer/Tracer.h"

using namespace std; 

int main(int argc, char **argv)
{
  if (argc<2) {
    fprintf(stderr, "USAGE: %s <input_file> <time_step>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string filename = argv[1];
  const int timestep = atoi(argv[2]);

  GLGPU3DDataset ds;
  ds.SetPrecomputeSupercurrent(true);
  ds.OpenDataFile(filename);
  ds.LoadTimeStep(timestep);

  FieldLineTracer tracer;
  tracer.SetDataset(&ds);
  tracer.Trace();
  tracer.WriteFieldLines(filename + ".trace.vtk");

  return 0;
}
