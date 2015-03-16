#include "VortexTransition.h"
#include <sstream>

VortexTransition::VortexTransition() :
  _num_global_vortices(0)
{
}

VortexTransition::~VortexTransition()
{
}

void VortexTransition::LoadFromFile(const std::string& dataname, int ts, int tl)
{
  for (int i=ts; i<ts+tl-1; i++) {
    std::stringstream ss;
    ss << dataname << ".match." << i << "." << i+1;

    VortexTransitionMatrix tm;
    if (tm.LoadFromFile(ss.str()))
      insert(std::make_pair(i, tm));
    else 
      fprintf(stderr, "cannot open file %s\n", ss.str().c_str());
  }
}

VortexTransitionMatrix VortexTransition::at(int i) const
{
  std::map<int, VortexTransitionMatrix>::const_iterator it = find(i);
  if (it != end())
    return it->second;
  else 
    return VortexTransitionMatrix();
}
