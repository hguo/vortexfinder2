#include "VortexLine.h"
#include "VortexLine.pb.h"

VortexLine::VortexLine() : 
  id(0), timestep(0)
{
}

VortexLine::~VortexLine()
{
}

///////
bool SerializeVortexLines(const std::vector<VortexLine>& lines, std::string& buf)
{
  PBVortexLines plines;
  for (int i=0; i<lines.size(); i++) {
    PBVortexLine *pline = plines.add_lines();
    for (int j=0; j<lines.size(); j++) 
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

