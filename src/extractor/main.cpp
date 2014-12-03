#include <iostream>
#include <vector>
#include "extractor.h"

using namespace libMesh; 

int main(int argc, char **argv)
{
  const std::string filename = "tslab.3.Bz0_02.Nt1000.lu.512.e"; 
  const double B[3] = {0.f, 0.f, 0.02f}; // magenetic field
  const double Kex = 0; 
  
  LibMeshInit init(argc, argv);
  
  VortexExtractor extractor(init.comm());
  // extractor.SetVerbose(1);
  extractor.SetMagneticField(B); 
  extractor.SetKex(Kex);
  extractor.SetGaugeTransformation(false);

  extractor.LoadData(filename);
  // for (int t=36; t<=40; t++) {
  {
    int t = 600; 
    fprintf(stderr, "------- timestep=%d -------\n", t); 
    
    double t0 = (double)clock() / CLOCKS_PER_SEC; 
    extractor.LoadTimestep(t);
    double t1 = (double)clock() / CLOCKS_PER_SEC; 
    extractor.Extract();
    double t2 = (double)clock() / CLOCKS_PER_SEC; 
    extractor.Trace(); 
    double t3 = (double)clock() / CLOCKS_PER_SEC; 

    fprintf(stderr, "------- timings -------\n");
    fprintf(stderr, "t_io:\t%f\n", t1-t0); 
    fprintf(stderr, "t_ex:\t%f\n", t2-t1); 
    fprintf(stderr, "t_tr:\t%f\n", t3-t2);
  }

  return 0; 
}
