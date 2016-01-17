#ifndef _STOCHASTIC_EXTRACTOR_H
#define _STOCHASTIC_EXTRACTOR_H

#include "Extractor.h"

class StochasticVortexExtractor : public VortexExtractor {
public:
  StochasticVortexExtractor();
  ~StochasticVortexExtractor();

  void SetNumberOfRuns(int);
  void SetKernelSize(float);
  void SetPertubation(float);

  void ExtractDeterministicVortices();
  void ExtractStochasticVortices();

  void EstimateDensities(int vid);

private:
  int _nruns;
  float _kernel_size;
  float _pertubation;
};

#endif
