#ifndef _STOCHASTIC_EXTRACTOR_H
#define _STOCHASTIC_EXTRACTOR_H

#include "Extractor.h"

class StochasticVortexExtractor : public VortexExtractor {
public:
  StochasticVortexExtractor();
  ~StochasticVortexExtractor();

  void SetNumberOfRuns(int);
  void SetNoiseAmplitude(float);
  void SetKernelSize(float);

  void ExtractDeterministicVortices();
  void ExtractStochasticVortices();

  void EstimateDensities(int vid);

private:
  int _nruns;
  float _noise_amplitude;
  float _kernel_size;
};

#endif
