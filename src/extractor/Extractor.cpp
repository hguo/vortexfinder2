#include "Extractor.h"

VortexExtractor::VortexExtractor()
{

}

VortexExtractor::~VortexExtractor()
{

}

void VortexExtractor::WriteVortexObjects(const std::string& filename)
{
  ::WriteVortexObjects(filename, _vortex_objects); 
}
