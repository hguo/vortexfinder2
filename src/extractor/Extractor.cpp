#include "Extractor.h"

VortexExtractor::VortexExtractor() :
  _gauge(false)
{

}

VortexExtractor::~VortexExtractor()
{

}

void VortexExtractor::SetGaugeTransformation(bool g)
{
  _gauge = g; 
}

void VortexExtractor::WriteVortexObjects(const std::string& filename)
{
  ::WriteVortexObjects(filename, _vortex_objects); 
}
