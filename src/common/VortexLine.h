#ifndef _VORTEX_LINE_H
#define _VORTEX_LINE_H

#include <string>
#include <list>
#include <vector>
#include <sstream>
#include "def.h"
#include "common/diy-ext.hpp"

/* 
 * \class   VortexLine
 * \author  Hanqi Guo
 * \brief   Vortex objects
*/
struct VortexLine : public std::vector<float>
{
  VortexLine();
  ~VortexLine(); 

  void ToBezier();
  void ToRegular(const float d=0.1);

  void Flattern(const float O[3], const float L[3]);
  void Unflattern(const float O[3], const float L[3]);

  void BoundingBox(float LB[3], float UB[3]) const;
  float MaxExtent() const;

  friend float MinimumDist(const VortexLine& l0, const VortexLine& l1);
  friend float CrossingPoint(const VortexLine& l0, const VortexLine& l1, float X[3]);
  friend float Area(const VortexLine& l0, const VortexLine& l1);

  int id, gid;
  int timestep;
  float time;
  bool is_bezier;
  bool is_loop;

  unsigned char r, g, b;
};

namespace diy {
  template <> struct Serialization<VortexLine> {
    static void save(diy::BinaryBuffer& bb, const VortexLine& m) {
      diy::save(bb, m.id);
      diy::save(bb, m.gid);
      diy::save(bb, m.timestep);
      diy::save(bb, m.time);
      diy::save(bb, m.is_bezier);
      diy::save(bb, m.is_loop);
      diy::save<std::vector<float> >(bb, m);
    }

    static void load(diy::BinaryBuffer&bb, VortexLine& m) {
      diy::load(bb, m.id);
      diy::load(bb, m.gid);
      diy::load(bb, m.timestep);
      diy::load(bb, m.time);
      diy::load(bb, m.is_bezier);
      diy::load(bb, m.is_loop);
      diy::load<std::vector<float> >(bb, m);
    }
  };
}

bool SaveVortexLinesVTK(const std::vector<VortexLine>& lines, const std::string& filename);

#endif
