#include <iostream>
#include "io/GLDatasetBase.h"
#include "extractor/Extractor.h"

int main(int argc, char **argv)
{
  if (argc<4) return EXIT_FAILURE;

  const std::string dataname = argv[1];
  const int T0 = atoi(argv[2]), 
            T = atoi(argv[3]);

  GLDatasetBase *ds = new GLDatasetBase;
  ds->SetDataName(dataname);
  if (!ds->LoadDefaultMeshGraph()) return EXIT_FAILURE;
  
  ds->SetTimeStep(T0, 0);
  ds->SetTimeStep(T0+1, 1);

  VortexExtractor *extractor = new VortexExtractor;
  extractor->SetDataset(ds);
#if 0
  fprintf(stderr, "...2\n");
  if (!extractor->LoadPuncturedFaces(0)) return EXIT_FAILURE;
  fprintf(stderr, "...3\n");
  if (!extractor->LoadPuncturedFaces(1)) return EXIT_FAILURE;
  fprintf(stderr, "...4\n");
  if (!extractor->LoadPuncturedEdges()) return EXIT_FAILURE;
  fprintf(stderr, "...5\n");
#endif

  fprintf(stderr, "tracing..\n");

  ds->SetTimeStep(T0);
  extractor->LoadPuncturedFaces(0);
  extractor->TraceOverSpace(0);
  extractor->SaveVortexLines(0);
  for (int t=T0+1; t<T0+T; t++) {
    ds->SetTimeStep(t, 1);
    extractor->LoadPuncturedFaces(1);
    extractor->TraceOverSpace(1);
    extractor->LoadPuncturedEdges();
    extractor->RelateOverTime();
    extractor->TraceOverTime();
    extractor->SaveVortexLines(1);
    extractor->RotateTimeSteps();
    ds->RotateTimeSteps();
  }

  // extractor->SaveVortexLines(dataname + ".vortex");

  delete extractor;
  delete ds; 

  return EXIT_SUCCESS;
}


#if 0
int main(int argc, char **argv)
{
  if (argc<4) return EXIT_FAILURE;

  const std::string dataname = argv[1];
  const int timestep = atoi(argv[2]), 
            timestep1 = atoi(argv[3]);

  GLDatasetBase *ds = new GLDatasetBase;
  ds->SetDataName(dataname);
  ds->SetTimeSteps(timestep, timestep1);
  if (!ds->LoadDefaultMeshGraph()) return EXIT_FAILURE;

  VortexExtractor *extractor = new VortexExtractor;
  extractor->SetDataset(ds);
  if (!extractor->LoadPuncturedFaces(0)) return EXIT_FAILURE;
  if (!extractor->LoadPuncturedFaces(1)) return EXIT_FAILURE;
  if (!extractor->LoadPuncturedEdges()) return EXIT_FAILURE;

  fprintf(stderr, "tracing..\n");
  extractor->TraceOverSpace(0);
  extractor->TraceOverSpace(1);
  extractor->RelateOverTime();
  extractor->TraceOverTime();
  // extractor->SaveVortexLines(dataname + ".vortex");

  delete extractor;
  delete ds; 

  return EXIT_SUCCESS;
}
#endif
