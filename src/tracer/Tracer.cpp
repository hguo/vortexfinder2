#include "Tracer.h"
#include "io/GLDataset.h"
#include "common/Utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <climits>

#if WITH_VTK
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>
#include <vtkXMLPolyDataWriter.h>
#endif

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
#if WITH_VTK
  vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::New();
  vtkSmartPointer<vtkPoints> points = vtkPoints::New();
  vtkSmartPointer<vtkCellArray> cells = vtkCellArray::New();

  int nv = 0;
  for (int i=0; i<_fieldlines.size(); i++) {
    const FieldLine& l = _fieldlines[i];
    
    vtkSmartPointer<vtkPolyLine> polyLine = vtkPolyLine::New();
    polyLine->GetPointIds()->SetNumberOfIds(l.size()/3);

    int j = 0;
    for (FieldLine::const_iterator it = l.begin(); it != l.end(); ) {
      double p[3] = {*(it++), *(it++), *(it++)};
      // double p[3] = {l[i*3], l[i*3+1], l[i*3+2]};
      points->InsertNextPoint(p);
      polyLine->GetPointIds()->SetId(j, j+nv);
      j ++;
    }
    cells->InsertNextCell(polyLine);
    nv += l.size()/3;
  }

  polyData->SetPoints(points);
  polyData->SetLines(cells);
  
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkXMLPolyDataWriter::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(polyData);
  writer->Write();
#else
  ::WriteFieldLines(filename, _fieldlines);
#endif
}

void FieldLineTracer::Trace()
{
  fprintf(stderr, "Trace..\n");

  // const int nseeds[3] = {8, 9, 8};
  const int nseeds[3] = {256, 128, 32};
  const float span[3] = {
    _ds->Lengths()[0]/(nseeds[0]-1), 
    _ds->Lengths()[1]/(nseeds[1]-1), 
    _ds->Lengths()[2]/(nseeds[2]-1)}; 

  for (int i=0; i<nseeds[0]; i++) {
    for (int j=0; j<nseeds[1]; j++) {
      for (int k=0; k<nseeds[2]; k++) {
        float seed[3] = {
          i * span[0] + _ds->Origins()[0], 
          j * span[1] + _ds->Origins()[1], 
          k * span[2] + _ds->Origins()[2]}; 
        Trace(seed);
      }
    }
  }
}

void FieldLineTracer::Trace(const float seed[3])
{
  static const int max_length = 2048; 
  const float h = 0.25; 
  float X[3] = {seed[0], seed[1], seed[2]}; 

  FieldLine line;

  fprintf(stderr, "Tracing line from X={%f, %f, %f}\n", seed[0], seed[1], seed[2]);

  // forward
  for (int n=0; n<max_length; n++) {
    line.push_back(X[0]); line.push_back(X[1]); line.push_back(X[2]); 
    // if (!RK4(X, h)) break;
    if (!RK1(X, h)) break;
  }
 
  // backward
  X[0] = seed[0]; X[1] = seed[1]; X[2] = seed[2];
  line.pop_front(); line.pop_front(); line.pop_front(); 
  for (int n=0; n<max_length; n++) {
    line.push_front(X[2]); line.push_front(X[1]); line.push_front(X[0]); 
    // if (!RK4(X, -h)) break;
    if (!RK1(X, -h)) break;
  }

  // fprintf(stderr, "length=%d\n", line.size()/3);
  if (line.size()/3 > 10)
    _fieldlines.push_back(line);
}

bool FieldLineTracer::Supercurrent(const float *X, float *J) const
{
  return (_ds->Supercurrent(X, J)); 
}

template <typename T>
bool FieldLineTracer::RK1(T *X, T h)
{
  T J[3]; 
  bool succ = Supercurrent(X, J);
  if (!succ) return false;

  const float threshold = 0.0001;
  float Jmag = sqrt(J[0]*J[0] + J[1]*J[1] + J[2]*J[2]);
  if (Jmag < threshold) return false;

  // fprintf(stderr, "X={%f, %f, %f}, J={%f, %f, %f}\n", 
  //     X[0], X[1], X[2], J[0], J[1], J[2]);

  X[0] = X[0] + h*J[0]; 
  X[1] = X[1] + h*J[1]; 
  X[2] = X[2] + h*J[2];

  return true; 
}

template <typename T>
bool FieldLineTracer::RK4(T *X, T h)
{
  T X0[3] = {X[0], X[1], X[2]};
  T J[3]; 
  
  // 1st RK step
  if (!Supercurrent(X, J)) return false;
  T k1[3]; 
  for (int i=0; i<3; i++) k1[i] = h * J[i];
  for (int i=0; i<3; i++) X[i] = X0[i] + 0.5 * k1[i];
  
  // 2nd RK step
  if (!Supercurrent(X, J)) return false;
  T k2[3]; 
  for (int i=0; i<3; i++) k2[i] = h * J[i];
  for (int i=0; i<3; i++) X[i] = X0[i] + 0.5 * k2[i];
  
  // 3rd RK step
  if (!Supercurrent(X, J)) return false;
  T k3[3]; 
  for (int i=0; i<3; i++) k3[i] = h * J[i];
  for (int i=0; i<3; i++) X[i] = X0[i] + k3[i];

  // 4th RK step
  if (!Supercurrent(X, J)) return false;
  for (int i=0; i<3; i++) 
    X[i] = X0[i] + (k1[i] + 2.0*(k2[i] + k3[i]) + h*J[i]) / 6.0;

  return true; 
}
