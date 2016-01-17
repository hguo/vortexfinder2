#include "StochasticExtractor.h"

StochasticVortexExtractor::StochasticVortexExtractor() :
  _nruns(256), 
  _noise_amplitude(0.04),
  _kernel_size(0.5)
{

}

StochasticVortexExtractor::~StochasticVortexExtractor()
{

}

void StochasticVortexExtractor::SetNumberOfRuns(int n)
{
  _nruns = n;
}

void StochasticVortexExtractor::SetNoiseAmplitude(float a)
{
  _noise_amplitude = a;
}

void StochasticVortexExtractor::SetKernelSize(float k)
{
  _kernel_size = k;
}

void StochasticVortexExtractor::ExtractDeterministicVortices()
{
  ExtractFaces(0);
  TraceOverSpace(0);
}

void StochasticVortexExtractor::ExtractStochasticVortices()
{
  // TODO
}
