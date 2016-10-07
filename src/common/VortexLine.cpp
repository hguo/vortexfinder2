#include "VortexLine.h"
#include "common/Utils.hpp"
#include "fitCurves/fitCurves.hpp"
#include "fitCurves/psimpl.h"
#include <climits>
#include <cfloat>
#include <cassert>

#if WITH_VTK
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>
#include <vtkXMLPolyDataWriter.h>
#endif

VortexLine::VortexLine() : 
  id(INT_MAX), 
  gid(INT_MAX), 
  timestep(0), 
  moving_speed(NAN),
  is_bezier(false), 
  is_loop(false)
{
}

VortexLine::~VortexLine()
{
}

void VortexLine::Print() const {
  for (int i=0; i<size()/3; i++) 
    fprintf(stderr, "(%f, %f, %f)\n", 
        at(i*3), at(i*3+1), at(i*3+2));
}

void VortexLine::Simplify(float tolorance)
{
  if (is_bezier) return;

  std::vector<float> R;
  psimpl::simplify_reumann_witkam<3>(begin(), end(), tolorance, std::back_inserter(R));
  // psimpl::simplify_douglas_peucker<3>(begin(), end(), tolorance, std::back_inserter(R));
  
  const int n0 = size()/3, n1 = R.size()/3;
  swap(R);
  // fprintf(stderr, "n0=%d, n1=%d\n", n0, n1);
}

void VortexLine::RemoveInvalidPoints() {
  if (is_bezier) return;

  std::vector<float> R;
  float lastPt[3];

  for (int i=0; i<size()/3; i++) {
    float currentPt[3] = {at(i*3), at(i*3+1), at(i*3+2)};

    bool valid = true;
    for (int j=0; j<3; j++) if (isnan(currentPt[j]) || isinf(currentPt[j])) valid = false;
    if (!valid) continue;
  
    if (i>1) {
      float d = dist(currentPt, lastPt);
      if (d>3) valid = false; // FIXME: arbitrary threshold
    }
    if (!valid) continue;

    memcpy(lastPt, currentPt, sizeof(float)*3);

    R.push_back(currentPt[0]);
    R.push_back(currentPt[1]);
    R.push_back(currentPt[2]);
  }
  swap(R);
}

void VortexLine::ToBezier(float error_bound)
{
  using namespace FitCurves;
  typedef Point<3> Pt;
  float tot_error;

  if (is_bezier) return;

  int npts = size()/3;
  Pt *pts = (Pt*)malloc(npts*3*sizeof(Pt));

  for (int i=0; i<npts; i++) {
    pts[i][0] = this->at(i*3);
    pts[i][1] = this->at(i*3+1);
    pts[i][2] = this->at(i*3+2);
  }

  Pt *pts1 = (Pt*)malloc(npts*4*sizeof(Pt));
  const int npts1 = fit_curves(npts, pts, error_bound, pts1, tot_error);

  clear();
  for (int i=0; i<npts1; i++) {
    push_back(pts1[i][0]);
    push_back(pts1[i][1]);
    push_back(pts1[i][2]);
  }

  free(pts);
  free(pts1);

  is_bezier = true;
}

float VortexLine::Length() const
{
  if (!length_acc.empty())
    return length_acc.back();

  length_seg.clear();
  length_acc.clear();

  const int npts = size()/3;
  float length = 0;
  for (int i=0; i<npts-1; i++) {
    const int j=i+1;
    float P0[3] = {at(i*3), at(i*3)+1, at(i*3)+2}, 
          P1[3] = {at(j*3), at(j*3)+1, at(j*3)+2};
    float d = dist(P0, P1);

    length_seg.push_back(d);
    length_acc.push_back(length);
    
    length += dist(P0, P1);
  }
  length_acc.push_back(length);
  return length;
}

bool VortexLine::Linear(float t, float X[3]) const 
{
  if (t <= 0) {
    X[0] = at(0); X[1] = at(1); X[2] = at(2); return true;
  } else if (t >= 1) {
    X[0] = at(size()-3); X[1] = at(size()-2); X[2] = at(size()-1); return true;
  }

  const float length = Length();
  const float tt = t * length;
  // fprintf(stderr, "length=%f, tt=%f\n", length, tt);
  int i0 = -1;

  assert(length_acc.size()>0);
  for (int i=0; i<length_acc.size()-1; i++) {
    if ((tt-length_acc[i]) * (tt-length_acc[i+1]) < 0) {
      i0 = i;
      break;
    }
  }

  if (i0 == -1) i0 = length_acc.size()-1;

  // assert(i0 != -1);
  // fprintf(stderr, "i0=%d, size=%d\n", i0, length_acc.size());
  assert(size()/3 == length_acc.size());
  const float tp = (tt-length_acc[i0])/length_seg[i0];
  for (int j=0; j<3; j++) {
    // X[j] = (1-tp) * at(i0*3+j) + tp * at((i0+1)*3+j);
    if (i0*3+j>=size()-1)
      fprintf(stderr, "%d, %d, %d, %d\n", i0*3+j, size(), i0, length_acc.size());
    X[j] = (1-tp) * at(i0*3+j) + tp * at(i0*3+j);
  }

  return true;
}

bool VortexLine::Bezier(float t, float X[3]) const // t \in [0, 1]
{
  using namespace FitCurves;
  typedef Point<3> Pt;
  
  if (!is_bezier) return false;
  if (t <= 0) {
    X[0] = at(0); X[1] = at(1); X[2] = at(2); return true;
  } else if (t >= 1) {
    X[0] = at(size()-3); X[1] = at(size()-2); X[2] = at(size()-1); return true;
  }

  const int npts = size()/3;
  const int nbs = npts/4;
  const int kInterval = t * nbs;
  const float tt = t * nbs - kInterval;

  Pt pts[4];
  for (int i=0; i<4; i++) {
    pts[i][0] = at((kInterval*4+i)*3);
    pts[i][1] = at((kInterval*4+i)*3+1);
    pts[i][2] = at((kInterval*4+i)*3+2);
  }
      
  Pt p = bezier(3, pts, tt);
  X[0] = p[0]; X[1] = p[1]; X[2] = p[2];

#if 0
  if (X[2]>10) {
    for (int i=0; i<4; i++)
      fprintf(stderr, "%f, %f, %f\n", pts[i][0], pts[i][1], pts[i][2]);
    fprintf(stderr, "t=%f, tt=%f, kInt=%d, val={%f, %f, %f}\n", t, tt, kInterval, X[0], X[1], X[2]);
  }
#endif
  return true;
}

void VortexLine::ToRegular(int N)
{
  length_seg.clear(); 
  length_acc.clear();

  const float delta = 1.f / (N - 1);
  std::vector<float> L;
  for (int i=0; i<N; i++) {
    float X[3];
    Bezier(i*delta, X);
    L.push_back(X[0]); L.push_back(X[1]); L.push_back(X[2]);
  }
  swap(L);
}

void VortexLine::ToRegularL(int N)
{
  const float delta = 1.f / (N - 1);
  std::vector<float> L;
  for (int i=0; i<N; i++) {
    float X[3];
    Linear(i*delta, X);
    L.push_back(X[0]); L.push_back(X[1]); L.push_back(X[2]);
  }
  swap(L);
}

void VortexLine::Flattern(const float O[3], const float L[3])
{
  int cross[3] = {0};
  const int n = size()/3;

  float p0[3], p[3];
  std::vector<float> line;

  for (int i=0; i<n; i++) {
    for (int j=0; j<3; j++) 
      p[j] = at(i*3+j) + cross[j] * L[j];
   
    if (i>0) 
      for (int j=0; j<3; j++) {
        if (p[j] - p0[j] > L[j]/2) {
          cross[j] --;
          p[j] -= L[j];
        } else if (p[j] - p0[j] < -L[j]/2) {
          cross[j] ++; 
          p[j] += L[j];
        }
      }

    for (int j=0; j<3; j++)
      line.push_back(p[j]);

    for (int j=0; j<3; j++) 
      p0[j] = p[j];
  }

  clear();
  swap(line);
}

void VortexLine::Unflattern(const float O[3], const float L[3])
{
  const int n = size()/3;
  float p[3];
  std::vector<float> line;

  for (int i=0; i<n; i++) {
    for (int j=0; j<3; j++) {
      p[j] = at(i*3+j);
      p[j] = fmod1(p[j] - O[j], L[j]) + O[j];
      line.push_back(p[j]);
    }
  }

  clear();
  swap(line);
}

float MinimumDist(const VortexLine& l0, const VortexLine& l1)
{
  float minDist = DBL_MAX;
  const int n0 = l0.size()/3, n1 = l1.size()/3;

  for (int i=0; i<n0; i++) 
    for (int j=0; j<n1; j++) {
      const float d[3] = {l0[i*3] - l1[j*3], l0[i*3+1] - l1[j*3+1], l0[i*3+2] - l1[j*3+2]};
      const float dist = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
      minDist = std::min(minDist, dist);
    }

  return minDist;
}

float CrossingPoint(const VortexLine& l0, const VortexLine& l1, float X[3])
{
  float minDist = DBL_MAX;
  const int n0 = l0.size()/3, n1 = l1.size()/3;
  int i0, j0;

  for (int i=0; i<n0; i++) 
    for (int j=0; j<n1; j++) {
      const float d[3] = {l0[i*3] - l1[j*3], l0[i*3+1] - l1[j*3+1], l0[i*3+2] - l1[j*3+2]};
      const float dist = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
      if (minDist > dist) {
        i0 = i;
        j0 = j;
        minDist = dist;
      }
    }

  X[0] = l0[i0*3] + l1[j0*3];
  X[1] = l0[i0*3+1] + l1[j0*3+1];
  X[2] = l0[i0*3+2] + l1[j0*3+2];

  return minDist;
}

float AreaL(const VortexLine& l0, const VortexLine& l1) 
{
  VortexLine b0 = l0, b1 = l1;
  b0.RemoveInvalidPoints();  b0.Simplify();  
  b1.RemoveInvalidPoints();  b1.Simplify();  

  const int N = 100;

  b0.ToRegularL(N);
  b1.ToRegularL(N);

  float a = 0;

  for (int i=0; i<N-1; i++) {
    const int j = i + 1;
    float A[3] = {b0[i*3], b0[i*3+1], b0[i*3+2]}, 
          B[3] = {b0[j*3], b0[j*3+1], b0[j*3+2]}, 
          C[3] = {b1[j*3], b1[j*3+1], b1[j*3+2]},
          D[3] = {b1[i*3], b1[i*3+1], b1[i*3+2]};
    float a1 = area(A, B, C) + area(A, C, D);
    a += a1;
#if 0  
    fprintf(stderr, "{%f, %f, %f}<->{%f, %f, %f}, a1=%f, a=%f\n", 
        b0[i*3], b0[i*3+1], b0[i*3+2], 
        b1[i*3], b1[i*3+1], b1[i*3+2], 
        a1, a);
#endif
  }

  return a;
}

float Area(const VortexLine& l0, const VortexLine& l1) 
{
  VortexLine b0 = l0, b1 = l1;
  b0.RemoveInvalidPoints();  b0.Simplify();  b0.ToBezier(); 
  b1.RemoveInvalidPoints();  b1.Simplify();  b1.ToBezier();

  const int N = 100;

  b0.ToRegular(N);
  b1.ToRegular(N);

  float a = 0;

  for (int i=0; i<N-1; i++) {
    const int j = i + 1;
    float A[3] = {b0[i*3], b0[i*3+1], b0[i*3+2]}, 
          B[3] = {b0[j*3], b0[j*3+1], b0[j*3+2]}, 
          C[3] = {b1[j*3], b1[j*3+1], b1[j*3+2]},
          D[3] = {b1[i*3], b1[i*3+1], b1[i*3+2]};
    float a1 = area(A, B, C) + area(A, C, D);
    a += a1;
#if 0  
    fprintf(stderr, "{%f, %f, %f}<->{%f, %f, %f}, a1=%f, a=%f\n", 
        b0[i*3], b0[i*3+1], b0[i*3+2], 
        b1[i*3], b1[i*3+1], b1[i*3+2], 
        a1, a);
#endif
  }

  return a;
}

void VortexLine::BoundingBox(float LB[3], float UB[3]) const
{
  LB[0] = LB[1] = LB[2] = FLT_MAX;
  UB[0] = UB[1] = UB[2] = -FLT_MAX;
  const VortexLine& l = *this;

  for (int i=0; i<size()/3; i++) {
    LB[0] = std::min(LB[0], l[i*3]);
    LB[1] = std::min(LB[1], l[i*3+1]);
    LB[2] = std::min(LB[2], l[i*3+2]);
    UB[0] = std::max(UB[0], l[i*3]);
    UB[1] = std::max(UB[1], l[i*3+1]);
    UB[2] = std::max(UB[2], l[i*3+2]);
  }
}

float VortexLine::MaxExtent() const
{
  float LB[3], UB[3];
  BoundingBox(LB, UB);

  float D[3] = {UB[0] - LB[0], UB[1] - LB[1], UB[2] - LB[2]};
  return std::max(std::max(D[0], D[1]), D[2]);
}

bool SaveVortexLinesVTK(const std::vector<VortexLine>& vlines, const std::string& filename)
{
#if WITH_VTK
  // FIXME: can only process 3D now
  vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::New();
  vtkSmartPointer<vtkPoints> points = vtkPoints::New();
  vtkSmartPointer<vtkCellArray> cells = vtkCellArray::New();
 
  std::vector<int> vertCounts;
  for (int i=0; i<vlines.size(); i++) {
    int vertCount = 0;
    const int nv = vlines[i].size()/3;
    // if (nv<2) continue;
    double p0[3];
    for (int j=0; j<nv; j++) {
      double p[3] = {vlines[i][j*3], vlines[i][j*3+1], vlines[i][j*3+2]};
      points->InsertNextPoint(p);

      double delta[3] = {p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
      double dist = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);

      if (j>0 && dist>5) { // FIXME
        vertCounts.push_back(vertCount);
        vertCount = 0;
      }
      memcpy(p0, p, sizeof(double)*3);
      vertCount ++;
    }

    if (vertCount > 0) 
      vertCounts.push_back(vertCount);
  }
    
  int nv = 0;
  for (int i=0; i<vertCounts.size(); i++) {
    // fprintf(stderr, "vertCount=%d\n", vertCounts[i]);
    vtkSmartPointer<vtkPolyLine> polyLine = vtkPolyLine::New();
    polyLine->GetPointIds()->SetNumberOfIds(vertCounts[i]);
    for (int j=0; j<vertCounts[i]; j++)
      polyLine->GetPointIds()->SetId(j, j+nv);

    cells->InsertNextCell(polyLine);
    nv += vertCounts[i];
  }

  polyData->SetPoints(points);
  polyData->SetLines(cells);
  
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkXMLPolyDataWriter::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(polyData);
  writer->Write();

  return true;
#else
  return false;
#endif
}
