#include "VortexLine.h"
#include "common/Utils.hpp"
#include "fitCurves/fitCurves.hpp"
#include <climits>
#include <cfloat>

#if WITH_PROTOBUF
#include "VortexLine.pb.h"
#endif

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
  is_bezier(false), 
  is_loop(false)
{
}

VortexLine::~VortexLine()
{
}

void VortexLine::ToBezier()
{
  using namespace FitCurves;
  typedef Point<3> Pt;
  const float error_bound = 0.001;
  float tot_error;

  if (is_bezier) return;

  int npts = size()/3-1;
  Pt *pts = (Pt*)malloc(npts*3*sizeof(Pt));

  for (int i=0; i<npts; i++) {
    pts[i][0] = this->at(i*3);
    pts[i][1] = this->at(i*3+1);
    pts[i][2] = this->at(i*3+2);
  }

  Pt *pts1 = (Pt*)malloc(npts*4*sizeof(Pt));
  int npts1 = fit_curves(npts, pts, error_bound, pts1, tot_error);

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

void VortexLine::ToRegular(const float stepsize)
{
  using namespace FitCurves;
  typedef Point<3> Pt;

  if (!is_bezier) return;
 
  int npts = size()/3;
  Pt *pts = (Pt*)malloc(npts*3*sizeof(Pt));

  for (int i=0; i<size()/3; i++) {
    pts[i][0] = at(i*3);
    pts[i][1] = at(i*3+1);
    pts[i][2] = at(i*3+2);
  }

  clear();
  
  for (int i=0; i<npts; i+=4) {
    const float tl = i<npts-5 ? 0.9999 : 1;
    for (float t=0; t<tl; t+=stepsize) {
      Pt p = bezier(3, pts+i, t);
      push_back(p[0]);
      push_back(p[1]);
      push_back(p[2]);
      // fprintf(stderr, "t=%f, p={%f, %f, %f}\n", t, p[0], p[1], p[2]);
    }
  }
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

///////
bool SerializeVortexLines(const std::vector<VortexLine>& lines, const std::string& info, std::string& buf)
{
#if WITH_PROTOBUF
  PBVortexLines plines;
  for (int i=0; i<lines.size(); i++) {
    PBVortexLine *pline = plines.add_lines();
    for (int j=0; j<lines[i].size(); j++) 
      pline->add_vertices( lines[i][j] );
    pline->set_id( lines[i].id );
    pline->set_timestep( lines[i].timestep );
    pline->set_time( lines[i].time );
    pline->set_bezier( lines[i].is_bezier );
  }
  if (info.length()>0) {
    plines.set_info_bytes(info);
  }
  return plines.SerializeToString(&buf);
#else
  return false;
#endif
}

bool UnserializeVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& buf)
{
#if WITH_PROTOBUF
  PBVortexLines plines;
  if (!plines.ParseFromString(buf)) return false;

  for (int i=0; i<plines.lines_size(); i++) {
    VortexLine line;
    for (int j=0; j<plines.lines(i).vertices_size(); j++) 
      line.push_back(plines.lines(i).vertices(j));
    line.id = plines.lines(i).id();
    line.timestep = plines.lines(i).timestep();
    line.time = plines.lines(i).time();
    line.is_bezier = plines.lines(i).bezier();
    lines.push_back(line);
  }

  if (plines.has_info_bytes())
    info = plines.info_bytes();

  return true;
#else
  return false;
#endif
}

bool SaveVortexLines(const std::vector<VortexLine>& lines, const std::string& info, const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) return false;

  std::string buf;
  SerializeVortexLines(lines, info, buf);
  fwrite(buf.data(), 1, buf.size(), fp);

  fclose(fp);
  return true;
}

bool LoadVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "rb"); 
  if (!fp) return false;

  fseek(fp, 0L, SEEK_END);
  size_t sz = ftell(fp);

  std::string buf;
  buf.resize(sz);
  fseek(fp, 0L, SEEK_SET);
  fread((char*)buf.data(), 1, sz, fp);
  fclose(fp);

  return UnserializeVortexLines(lines, info, buf);
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
