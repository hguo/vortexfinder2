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
  extractor.LoadTimestep(600); 

  extractor.Extract();
  extractor.Trace(); 

  return 0; 
}
