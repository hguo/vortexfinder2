#include <iostream>
#include "io/Condor2Dataset.h"
#include "Tracer.h"

using namespace std; 

int main(int argc, char **argv)
{
  if (argc<3) {
    fprintf(stderr, "USAGE: %s <input_file> <time_step>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string filename = argv[1];
  const int timestep = atoi(argv[2]);
  
  libMesh::LibMeshInit init(1, argv); // set argc to 1 to supress PETSc warnings. 
  Condor2Dataset ds(init.comm()); 
  
  ds.OpenDataFile(filename);
  ds.LoadTimeStep(timestep);

  FieldLineTracer tracer;
  tracer.SetDataset(&ds);
  tracer.Trace();
  tracer.WriteFieldLines(filename + ".trace");

  return 0;
}
