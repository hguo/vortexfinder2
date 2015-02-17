#include <iostream>
#include <vector>
#include <getopt.h>
#include "io/GLGPU3DDataset.h"
#include "extractor/GLGPU3DExtractor.h"

static std::string filename_in, filename_out;
static int nogauge = 0,  
           verbose = 0, 
           benchmark = 0; 

static struct option longopts[] = {
  {"verbose", no_argument, &verbose, 1},  
  {"nogauge", no_argument, &nogauge, 1},
  {"benchmark", no_argument, &benchmark, 1}, 
  {"input", required_argument, 0, 'i'},
  {"output", required_argument, 0, 'o'},
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
  if (!ds.Valid()) {
    fprintf(stderr, "Invalid input data.\n");
    return EXIT_FAILURE;
  }
  ds.PrintInfo();
  
  GLGPU3DVortexExtractor extractor;
  extractor.SetDataset(&ds);
  // extractor.SetVerbose(verbose);
  extractor.SetGaugeTransformation(!nogauge);
  
  extractor.Extract();
  // extractor.Trace(); 
  // extractor.WriteVortexObjects(filename_out); 

  return EXIT_SUCCESS; 
}
