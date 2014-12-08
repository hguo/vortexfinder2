#include "gldataset.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

GLDataset::GLDataset()
  : _Jx(0), _Kex(0), _Kex_dot(0), _fluctuation_amp(0)
{
  memset(_lengths, 0, sizeof(double)*3); 
  memset(_B, 0, sizeof(double)*3); 
}

GLDataset::~GLDataset()
{
}
