#ifndef _VORTEX_TRANSITION_H
#define _VORTEX_TRANSITION_H

#include "common/VortexTransitionMatrix.h"

class VortexTransition : public std::map<int, VortexTransitionMatrix> 
{
public:
  VortexTransition();
  ~VortexTransition();

private:
  int _num_global_vortices;
};

#endif
