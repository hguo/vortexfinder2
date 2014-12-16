#include <iostream>
#include "io/GLGPUDataset.h"
#include "Tracer.h"

using namespace std; 

int main(int argc, char **argv)
{
  if (argc<2) {
    fprintf(stderr, "USAGE: %s <input_file>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string filename = argv[1]; 

  GLGPUDataset ds;
  ds.OpenDataFile(filename);
  ds.ComputeSupercurrentField();

  FieldLineTracer tracer;
  tracer.SetDataset(&ds);
  tracer.Trace();
  tracer.WriteFieldLines(filename + ".trace");

  return 0;
}
