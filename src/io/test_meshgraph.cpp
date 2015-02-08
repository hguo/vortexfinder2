#include "io/Condor2Dataset.h"

int main(int argc, char **argv)
{
  if (argc<2) return 1;

  libMesh::LibMeshInit init(1, argv); // set argc to 1 to supress PETSc warnings. 

  Condor2Dataset ds(init.comm()); 
  ds.OpenDataFile(argv[1]);
  
  return 0;
}
