#include "StochasticExtractor.h"

StochasticVortexExtractor::StochasticVortexExtractor() :
  _nruns(256), 
  _kernel_size(0.5),
  _pertubation(0.04)
{

}

StochasticVortexExtractor::~StochasticVortexExtractor()
{

}

void StochasticVortexExtractor::SetNumberOfRuns(int n)
{
  _nruns = n;
}

void StochasticVortexExtractor::SetKernelSize(float k)
{
  _kernel_size = k;
}

void StochasticVortexExtractor::SetPertubation(float p)
{
  _pertubation = p;
}

void StochasticVortexExtractor::ExtractDeterministicVortices()
{
  Clear();
  VortexExtractor::SetPertubation(0);
  ExtractFaces(0);
  TraceOverSpace(0);
}

void StochasticVortexExtractor::ExtractStochasticVortices()
{
  for (int i=0; i<_nruns; i++) {
    Clear();
    VortexExtractor::SetPertubation(_pertubation);
    ExtractFaces(0);
    TraceOverSpace(0);
  }
}
