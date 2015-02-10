#include "Puncture.h"
#include "common/Puncture.pb.h"

bool SerializePuncturedFaces(const std::map<FaceIdType, PuncturedFace> m, std::string &buf)
{
  PBPuncturedFaces pfaces;
  for (std::map<FaceIdType, PuncturedFace>::const_iterator it = m.begin(); it != m.end(); it ++) {
    PBPuncturedFace *pface = pfaces.add_faces();
    pface->set_id( it->first );
    pface->set_chirality( it->second.chirality );
    pface->set_x( it->second.pos[0] );
    pface->set_y( it->second.pos[1] );
    pface->set_z( it->second.pos[2] );
  }
  return pfaces.SerializeToString(&buf);
}

bool UnserializePuncturedFaces(std::map<FaceIdType, PuncturedFace> &m, const std::string &buf)
{
  PBPuncturedFaces pfaces;
  if (!pfaces.ParseFromString(buf)) return false;

  m.clear();
  for (int i=0; i<pfaces.faces_size(); i++) {
    PuncturedFace face;
    face.chirality = pfaces.faces(i).chirality();
    face.pos[0] = pfaces.faces(i).x();
    face.pos[1] = pfaces.faces(i).y();
    face.pos[2] = pfaces.faces(i).z();

    m[pfaces.faces(i).id()] = face;
  }
  return true;
}

bool SavePuncturedFaces(const std::map<FaceIdType, PuncturedFace> m, const std::string &filename)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) return false;

  std::string buf;
  SerializePuncturedFaces(m, buf);
  fwrite(buf.data(), 1, buf.size(), fp);

  fclose(fp);
  return true;
}

bool LoadPuncturedFaces(std::map<FaceIdType, PuncturedFace> &m, const std::string &filename)
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

  return UnserializePuncturedFaces(m, buf);
}


//////// I/O for edges
bool SerializePuncturedEdges(const std::map<EdgeIdType, PuncturedEdge> m, std::string &buf)
{
  PBPuncturedEdges pedges;
  for (std::map<EdgeIdType, PuncturedEdge>::const_iterator it = m.begin(); it != m.end(); it ++) {
    PBPuncturedEdge *pedge = pedges.add_edges();
    pedge->set_id( it->first );
    pedge->set_chirality( it->second.chirality );
    pedge->set_t( it->second.t );
  }
  return pedges.SerializeToString(&buf);
}

bool UnserializePuncturedEdges(std::map<EdgeIdType, PuncturedEdge> &m, const std::string &buf)
{
  PBPuncturedEdges pedges;
  if (!pedges.ParseFromString(buf)) return false;

  for (int i=0; i<pedges.edges_size(); i++) {
    PuncturedEdge edge;
    edge.chirality = pedges.edges(i).chirality();
    edge.t = pedges.edges(i).t();

    m[pedges.edges(i).id()] = edge;
  }
  return true;
}

bool SavePuncturedEdges(const std::map<EdgeIdType, PuncturedEdge> m, const std::string &filename)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) return false;

  std::string buf;
  SerializePuncturedEdges(m, buf);
  fwrite(buf.data(), 1, buf.size(), fp);

  fclose(fp);
  return true;
}

bool LoadPuncturedEdges(std::map<EdgeIdType, PuncturedEdge> &m, const std::string &filename)
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

  return UnserializePuncturedEdges(m, buf);
}
