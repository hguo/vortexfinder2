#include "StochasticExtractor.h"
#include "vfgpu/vfgpu.h"

StochasticVortexExtractor::StochasticVortexExtractor() :
  _nruns(256), 
  _kernel_size(0.5),
  _pertubation(0.04),
  _density(NULL)
{

}

StochasticVortexExtractor::~StochasticVortexExtractor()
{
  if (_density != NULL)
    free(_density);
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
  std::vector<float> pts;
  std::vector<int> acc;
  int count = 0;
  
  for (int i=0; i<_nruns; i++) {
    Clear();
    VortexExtractor::SetPertubation(_pertubation);
    ExtractFaces(0);
    TraceOverSpace(0);
    VortexObjectsToVortexLines(0);

    for (int i=0; i<_vortex_lines.size(); i++) {
      const VortexLine& l = _vortex_lines[i];
      count += l.size();
      for (int j=0; j<l.size(); j++) 
        pts.push_back(l[j]);
      acc.push_back(count);
    }
  }

  typedef std::chrono::high_resolution_clock clock;
  auto t0 = clock::now();
  
  // vfgpu_density_estimate(pts.size()/3, acc.size(), pts.data(), acc.data());

  auto t1 = clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000000000.0; 
  fprintf(stderr, "t_density=%f\n", elapsed);
}
