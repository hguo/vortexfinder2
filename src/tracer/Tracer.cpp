#include "Tracer.h"
#include "io/GLDataset.h"

FieldLineTracer::FieldLineTracer()
{

}

FieldLineTracer::~FieldLineTracer()
{

}

void FieldLineTracer::SetDataset(const GLDataset* ds)
{
  _ds = ds;
}

void FieldLineTracer::WriteFieldLines(const std::string& filename)
{
  ::WriteFieldLines(filename, _fieldlines);
}

void FieldLineTracer::Trace()
{
  fprintf(stderr, "Trace..\n");

  const int nseeds[3] = {8, 9, 8};
  const double span[3] = {
    _ds->Lengths()[0]/(nseeds[0]-1), 
    _ds->Lengths()[1]/(nseeds[1]-1), 
    _ds->Lengths()[2]/(nseeds[2]-1)}; 

  for (int i=0; i<nseeds[0]; i++) {
    for (int j=0; j<nseeds[1]; j++) {
      for (int k=0; k<nseeds[2]; k++) {
        double seed[3] = {
          i * span[0] + _ds->Origins()[0], 
          j * span[1] + _ds->Origins()[1], 
          k * span[2] + _ds->Origins()[2]}; 
        Trace(seed);
      }
    }
  }
}

void FieldLineTracer::Trace(const double seed[3])
{
  static const int max_length = 1024; 
  const double h = 0.02; 
  double pt[3] = {seed[0], seed[1], seed[2]}; 

  std::list<double> line;

  // forward
  for (int n=0; n<max_length; n++) {
    line.push_back(pt[0]); line.push_back(pt[1]); line.push_back(pt[2]); 
    if (!RK1(pt, h)) break;
  }
 
  // backward
  pt[0] = seed[0]; pt[1] = seed[1]; pt[2] = seed[2];
  line.pop_front(); line.pop_front(); line.pop_front(); 
  for (int n=0; n<max_length; n++) {
    line.push_front(pt[2]); line.push_front(pt[1]); line.push_front(pt[0]); 
    if (!RK1(pt, -h)) break;
  }

  FieldLine line1(line);
  _fieldlines.push_back(line1);
}

bool FieldLineTracer::RK1(double *pt, double h)
{
  double J[3]; 
  if (!_ds->Supercurrent(pt, J)) return false;

  pt[0] = pt[0] + h*J[0]; 
  pt[1] = pt[1] + h*J[1]; 
  pt[2] = pt[2] + h*J[2];

  return true; 
}
