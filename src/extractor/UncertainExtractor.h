#ifndef _UNCERTAIN_EXTRACTOR_H
#define _UNCERTAIN_EXTRACTOR_H

#include "Extractor.h"

class UncertainVortexExtractor {
public:
  UncertainVortexExtractor();
  ~UncertainVortexExtractor();

  void SetNumberOfRuns(int);

  void ExtractDeterministicVortices();
  void ExtractStochasticVortices();

  void EstimateDensities(int vid);

private:
  int _nruns;
};

#endif
