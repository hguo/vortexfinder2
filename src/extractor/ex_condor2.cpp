#include <iostream>
#include <vector>
#include <getopt.h>
#include "io/Condor2Dataset.h"
#include "extractor/Condor2Extractor.h"

static std::string filename_in, filename_out;
static int nogauge = 0,  
           verbose = 0, 
           benchmark = 0; 
static int T0=1, T=1; // start and length of timesteps

static struct option longopts[] = {
  {"verbose", no_argument, &verbose, 1},  
  {"nogauge", no_argument, &nogauge, 1},
  {"benchmark", no_argument, &benchmark, 1}, 
  {"input", required_argument, 0, 'i'},
  {"output", required_argument, 0, 'o'},
  {"time", required_argument, 0, 't'}, 
  {"length", required_argument, 0, 'l'}, 
  {0, 0, 0, 0} 
};

static bool parse_arg(int argc, char **argv)
{
  int c; 

  while (1) {
    int option_index = 0;
    c = getopt_long(argc, argv, "i:o:t:l", longopts, &option_index); 
    if (c == -1) break;

    switch (c) {
    case 'i': filename_in = optarg; break;
    case 'o': filename_out = optarg; break;
    case 't': T0 = atoi(optarg); break;
    case 'l': T = atoi(optarg); break;
    default: break; 
    }
  }

  if (optind < argc) {
    if (filename_in.empty())
      filename_in = argv[optind++]; 
  }

  if (filename_in.empty()) {
    fprintf(stderr, "FATAL: input filename not given.\n"); 
    return false;
  } 
  
  if (filename_out.empty()) 
    filename_out = filename_in + ".vortex"; 

  if (verbose) {
    fprintf(stderr, "---- Argument Summary ----\n"); 
    fprintf(stderr, "filename_in=%s\n", filename_in.c_str()); 
    fprintf(stderr, "filename_out=%s\n", filename_out.c_str()); 
    fprintf(stderr, "nogauge=%d\n", nogauge);
    fprintf(stderr, "t=%d\n", T0);
    fprintf(stderr, "T=%d\n", T);
    fprintf(stderr, "--------------------------\n"); 
  }

  return true;  
}

static void print_help(int argc, char **argv)
{
  fprintf(stderr, "USAGE:\n");
  fprintf(stderr, "%s -i <input_filename> [-o output_filename] [--nogauge] [-t=<t>] [-T=<T>] [-Kx=<Kx>] [-Bx=<Bx>] [-By=<By>] [-Bz=<Bz>]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "\t--verbose   verbose output\n"); 
  fprintf(stderr, "\t--benchmark Enable benchmark\n"); 
  fprintf(stderr, "\t--nogauge   Disable gauge transformation\n"); 
  fprintf(stderr, "\t--Kx        Kx\n");
  fprintf(stderr, "\t-t          Starting time step for the analysis\n"); 
  fprintf(stderr, "\t-T          Number of time step for the analysis\n"); 
  fprintf(stderr, "\n");
}

int main(int argc, char **argv)
{
  if (!parse_arg(argc, argv)) {
    print_help(argc, argv);
    return EXIT_FAILURE;
  }

  // libMesh::LibMeshInit init(argc, argv);
  libMesh::LibMeshInit init(1, argv); // set argc to 1 to supress PETSc warnings. 
 
  Condor2Dataset ds(init.comm()); 
  ds.SetKex(0);
  ds.PrintInfo();
  
  ds.OpenDataFile(filename_in);
  if (!ds.Valid()) {
    fprintf(stderr, "Invalid input data.\n");
    return EXIT_FAILURE;
  }
  
  Condor2VortexExtractor extractor;
  extractor.SetDataset(&ds);
  extractor.SetGaugeTransformation(!nogauge);

  ds.LoadTimeStep(T0);
  extractor.ExtractFaces(0);
  for (int t=T0+1; t<T0+T; t++) {
    ds.LoadTimeStep1(t);
    extractor.ExtractEdges();
    extractor.ExtractFaces(1);
    // extractor.TraceOverTime(); 
    // extractor.TraceVirtualCells(); 
    // extractor.SaveVortexLines(filename_out); 
  }

  return EXIT_SUCCESS; 
}
