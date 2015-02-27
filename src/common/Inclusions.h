#ifndef _INCLUSIONS_H
#define _INCLUSIONS_H

#include <vector>

class Inclusions {
public:
  Inclusions();
  ~Inclusions();

  void PrintInfo();

  void Clear();
  int Count() const {return _x.size();}

  double Radius() const {return _radius;}
  double x(int i) const {return _x[i];}
  double y(int i) const {return _y[i];}
  double z(int i) const {return _z[i];}

  bool ParseFromTextFile(const std::string& filename);

  bool SerializeToString(std::string& str);
  bool UnserializeToString(const std::string& str);

protected:
  double _radius;
  std::vector<double> _x, _y, _z;
};

#endif
