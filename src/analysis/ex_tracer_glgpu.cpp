#include <iostream>
#include "io/GLGPU3DDataset.h"
#include "tracer/Tracer.h"

using namespace std; 

int main(int argc, char **argv)
{
  if (argc<2) {
    fprintf(stderr, "USAGE: %s <input_file>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string filename = argv[1]; 

  GLGPU3DDataset ds;
  ds.SetPrecomputeSupercurrent(true);
  ds.OpenDataFile(filename);

  FieldLineTracer tracer;
  tracer.SetDataset(&ds);
  tracer.Trace();
  tracer.WriteFieldLines(filename + ".trace");

  return 0;
}
