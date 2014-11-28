#include "vortex.h"

VortexObject::VortexObject()
{
}

VortexObject::~VortexObject()
{
}

void VortexObject::SerializeToString(std::string& str)
{
  // TODO
}

bool VortexObject::ParseFromString(const std::string& str)
{
  // TODO
}

int VortexObject::AddLine() 
{
  return AddLine(std::list<float>()); 
}

int VortexObject::AddLine(const std::list<float> &line)
{
  push_back(line);  
  return size() - 1; 
}
