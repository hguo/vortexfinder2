#include <iostream>
#include <vector>
#include <getopt.h>
#include "io/Condor2Dataset.h"
#include "extractor/Condor2Extractor.h"

static std::string filename_in, filename_out;
static double Kex = 0;
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
  {"Kx", required_argument, 0, 'k'}, 
  {"t", required_argument, 0, 't'}, 
  {"T", required_argument, 0, 'T'}, 
  {0, 0, 0, 0} 
};

static bool parse_arg(int argc, char **argv)
{
  int c; 

  while (1) {
    int option_index = 0;
    c = getopt_long(argc, argv, "i:o:k:x:y:z:t:T", longopts, &option_index); 
    if (c == -1) break;

    switch (c) {
    case 'i': filename_in = optarg; break;
    case 'o': filename_out = optarg; break;
    case 'k': Kex = atof(optarg); break;
    case 't': T0 = atoi(optarg); break;
    case 'T': T = atoi(optarg); break;
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
    fprintf(stderr, "Kex=%f\n", Kex); 
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
  ds.SetKex(Kex);
  ds.PrintInfo();
  
  ds.OpenDataFile(filename_in);
  if (!ds.Valid()) {
    fprintf(stderr, "Invalid input data.\n");
    return EXIT_FAILURE;
  }
  
  Condor2VortexExtractor extractor;
  extractor.SetDataset(&ds);
  // extractor.SetVerbose(verbose);
  extractor.SetGaugeTransformation(!nogauge);

  for (int t=T0; t<T0+T; t++) {
    fprintf(stderr, "Analyzing timestep %d...\n", t); 
    
    double t0 = (double)clock() / CLOCKS_PER_SEC; 
    ds.LoadTimeStep(t);
    double t1 = (double)clock() / CLOCKS_PER_SEC; 
    extractor.Extract();
    double t2 = (double)clock() / CLOCKS_PER_SEC; 
    extractor.Trace(); 
    double t3 = (double)clock() / CLOCKS_PER_SEC; 
    extractor.WriteVortexObjects(filename_out); 

    if (benchmark) {
      fprintf(stderr, "------- timings -------\n");
      fprintf(stderr, "t_io:\t%f\n", t1-t0); 
      fprintf(stderr, "t_ex:\t%f\n", t2-t1); 
      fprintf(stderr, "t_tr:\t%f\n", t3-t2);
    }
  }

  return EXIT_SUCCESS; 
}
