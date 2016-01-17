#ifndef _STOCHASTIC_EXTRACTOR_H
#define _STOCHASTIC_EXTRACTOR_H

#include "Extractor.h"

class StochasticVortexExtractor {
public:
  StochasticVortexExtractor();
  ~StochasticVortexExtractor();

  void SetNumberOfRuns(int);

  void ExtractDeterministicVortices();
  void ExtractStochasticVortices();

  void EstimateDensities(int vid);

private:
  int _nruns;
};

#endif
