#include "VortexLine.h"
#include "VortexLine.pb.h"
#include "common/Utils.hpp"
#include "fitCurves/fitCurves.hpp"
#include <climits>

VortexLine::VortexLine() : 
  id(INT_MAX), gid(INT_MAX), timestep(0), is_bezier(false)
{
}

VortexLine::~VortexLine()
{
}

void VortexLine::ToBezier()
{
  using namespace FitCurves;
  typedef Point<3> Pt;
  const double error_bound = 0.001;
  double tot_error;

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

void VortexLine::ToRegular(const double stepsize)
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
    const double tl = i<npts-5 ? 0.9999 : 1;
    for (double t=0; t<tl; t+=stepsize) {
      Pt p = bezier(3, pts+i, t);
      push_back(p[0]);
      push_back(p[1]);
      push_back(p[2]);
      // fprintf(stderr, "t=%f, p={%f, %f, %f}\n", t, p[0], p[1], p[2]);
    }
  }
}

void VortexLine::Flattern(const double O[3], const double L[3])
{
  int cross[3] = {0};
  const int n = size()/3;

  double p0[3], p[3];
  std::vector<double> line;

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

void VortexLine::Unflattern(const double O[3], const double L[3])
{
  const int n = size()/3;
  double p[3];
  std::vector<double> line;

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
  PBVortexLines plines;
  for (int i=0; i<lines.size(); i++) {
    PBVortexLine *pline = plines.add_lines();
    for (int j=0; j<lines[i].size(); j++) 
      pline->add_vertices( lines[i][j] );
    pline->set_id( lines[i].id );
    pline->set_timestep( lines[i].timestep );
    pline->set_bezier( lines[i].is_bezier );
  }
  if (info.length()>0) {
    plines.set_info_bytes(info);
  }
  return plines.SerializeToString(&buf);
}

bool UnserializeVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& buf)
{
  PBVortexLines plines;
  if (!plines.ParseFromString(buf)) return false;

  for (int i=0; i<plines.lines_size(); i++) {
    VortexLine line;
    for (int j=0; j<plines.lines(i).vertices_size(); j++) 
      line.push_back(plines.lines(i).vertices(j));
    line.id = plines.lines(i).id();
    line.timestep = plines.lines(i).timestep();
    line.is_bezier = plines.lines(i).bezier();
    lines.push_back(line);
  }

  if (plines.has_info_bytes())
    info = plines.info_bytes();

  return true;
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

