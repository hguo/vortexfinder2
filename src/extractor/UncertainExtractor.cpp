#include "UncertainExtractor.h"

UncertainVortexExtractor::UncertainVortexExtractor() :
  _nruns(256)
{

}

UncertainVortexExtractor::~UncertainVortexExtractor()
{

}

void UncertainVortexExtractor::SetNumberOfRuns(int n)
{
  _nruns = n;
}

void UncertainVortexExtractor::ExtractDeterministicVortices()
{
  // TODO
}

void UncertainVortexExtractor::ExtractStochasticVortices()
{
  // TODO
}
