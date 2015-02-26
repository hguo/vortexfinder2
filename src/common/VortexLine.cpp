#include "VortexLine.h"
#include "VortexLine.pb.h"
#include "fitCurves/fitCurves.hpp"

VortexLine::VortexLine() : 
  id(0), timestep(0), is_bezier(false)
{
}

VortexLine::~VortexLine()
{
}

void VortexLine::RegularToBezier()
{
  using namespace FitCurves;
  typedef Point<3> Pt;
  const double error_bound = 0.2;
  double tot_error;

  if (is_bezier) return;

  int npts = size()/3;
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

void VortexLine::BezierToRegular()
{
#if 0
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

  double t = 0;
  while (1) {
    bezier(3, pts, t);
  }
#endif
}

///////
bool SerializeVortexLines(const std::vector<VortexLine>& lines, std::string& buf)
{
  PBVortexLines plines;
  for (int i=0; i<lines.size(); i++) {
    PBVortexLine *pline = plines.add_lines();
    for (int j=0; j<lines[i].size(); j++) 
      pline->add_vertices( lines[i][j] );
    pline->set_id( lines[i].id );
    pline->set_timestep( lines[i].timestep );
  }
  return plines.SerializeToString(&buf);
}

bool UnserializeVortexLines(std::vector<VortexLine>& lines, const std::string& buf)
{
  PBVortexLines plines;
  if (!plines.ParseFromString(buf)) return false;

  for (int i=0; i<plines.lines_size(); i++) {
    VortexLine line;
    for (int j=0; j<plines.lines(i).vertices_size(); j++) 
      line.push_back(plines.lines(i).vertices(j));
    line.id = plines.lines(i).id();
    line.timestep = plines.lines(i).timestep();
    lines.push_back(line);
  }

  return true;
}

bool SaveVortexLines(const std::vector<VortexLine>& lines, const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) return false;

  std::string buf;
  SerializeVortexLines(lines, buf);
  fwrite(buf.data(), 1, buf.size(), fp);

  fclose(fp);
  return true;
}

bool LoadVortexLines(std::vector<VortexLine>& lines, const std::string& filename)
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

  return UnserializeVortexLines(lines, buf);
}

