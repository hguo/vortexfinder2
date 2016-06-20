#ifndef _INCLUSIONS_H
#define _INCLUSIONS_H

#include <vector>
#include <string>

class Inclusions {
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

#endif
