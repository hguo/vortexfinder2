#include <iostream>
#include <vector>
#include <getopt.h>
#include "io/GLGPU3DDataset.h"
#include "extractor/GLGPUExtractor.h"

static std::string filename_in, filename_out;
static int nogauge = 0,  
           verbose = 0, 
           benchmark = 0; 
static int T0=0, T=1; // start and length of timesteps
static int span=1;

static struct option longopts[] = {
  {"verbose", no_argument, &verbose, 1},  
  {"nogauge", no_argument, &nogauge, 1},
  {"benchmark", no_argument, &benchmark, 1}, 
  {"input", required_argument, 0, 'i'},
  {"output", required_argument, 0, 'o'},
  {"time", required_argument, 0, 't'}, 
  {"length", required_argument, 0, 'l'},
  {"span", required_argument, 0, 's'},
  {0, 0, 0, 0} 
};

static bool parse_arg(int argc, char **argv)
{
  int c; 

  while (1) {
    int option_index = 0;
    c = getopt_long(argc, argv, "i:o:t:l:s", longopts, &option_index); 
    if (c == -1) break;

    switch (c) {
    case 'i': filename_in = optarg; break;
    case 'o': filename_out = optarg; break;
    case 't': T0 = atoi(optarg); break;
    case 'l': T = atoi(optarg); break;
    case 's': span = atoi(optarg); break;
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
    fprintf(stderr, "--------------------------\n"); 
  }

  return true;  
}

static void print_help(int argc, char **argv)
{
  fprintf(stderr, "USAGE:\n");
  fprintf(stderr, "%s -i <input_filename> [-o output_filename] [--nogauge]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "\t--verbose   verbose output\n"); 
  fprintf(stderr, "\t--benchmark Enable benchmark\n"); 
  fprintf(stderr, "\t--nogauge   Disable gauge transformation\n"); 
  fprintf(stderr, "\n");
}


int main(int argc, char **argv)
{
  if (!parse_arg(argc, argv)) {
    print_help(argc, argv);
    return EXIT_FAILURE;
  }

  GLGPU3DDataset ds;
  ds.OpenDataFile(filename_in);
  ds.LoadTimeStep(T0, 0);
  ds.BuildMeshGraph();
  ds.PrintInfo();
 
  GLGPUVortexExtractor extractor;
  extractor.SetDataset(&ds);
  extractor.SetGaugeTransformation(!nogauge);
  
  extractor.ExtractFaces(0);
  extractor.TraceOverSpace(0);
  extractor.SaveVortexLines(0);
  for (int t=T0+span; t<T0+T; t+=span){
    ds.LoadTimeStep(t, 1);
    // ds.PrintInfo(1);
    extractor.ExtractFaces(1);
    extractor.TraceOverSpace(1);
    extractor.ExtractEdges();
    extractor.TraceOverTime();
    extractor.SaveVortexLines(1);
    extractor.RotateTimeSteps();
    ds.RotateTimeSteps();
  }

  return EXIT_SUCCESS; 
}
