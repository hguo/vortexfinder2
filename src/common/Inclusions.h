#ifndef _INCLUSIONS_H
#define _INCLUSIONS_H

#include <vector>
#include <string>
#include "def.h"

class Inclusions {
  // friend class diy::Serialization<Inclusions>;
public:
  Inclusions();
  ~Inclusions();

  void PrintInfo();

  void Clear();
  int Count() const {return _x.size();}

  float Radius() const {return _radius;}
  float x(int i) const {return _x[i];}
  float y(int i) const {return _y[i];}
  float z(int i) const {return _z[i];}

  bool ParseFromTextFile(const std::string& filename);

  bool SerializeToString(std::string& str);
  bool UnserializeToString(const std::string& str);

protected:
  float _radius;
  std::vector<float> _x, _y, _z;
};

#if 0
namespace diy {
  template <> struct Serialization<Inclusions> {
    static void save(diy::BinaryBuffer& bb, const Inclusions& m) {
      diy::save(bb, m._radius);
      diy::save(bb, m._x);
      diy::save(bb, m._y);
      diy::save(bb, m._z);
    }

    static void load(diy::BinaryBuffer&bb, Inclusions& m) {
      diy::load(bb, m._radius);
      diy::load(bb, m._x);
      diy::load(bb, m._y);
      diy::load(bb, m._z);
    }
  };
}
#endif

#endif
