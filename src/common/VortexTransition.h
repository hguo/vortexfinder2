#ifndef _VORTEX_TRANSITION_H
#define _VORTEX_TRANSITION_H

#include <string>
#include <vector>

class VortexTransitionMatrix {
public:
  VortexTransitionMatrix();
  VortexTransitionMatrix(int t0, int t1, int n0, int n1);
  ~VortexTransitionMatrix();

  bool LoadFromFile(const std::string& filename);
  bool SaveToFile(const std::string& filename) const;

  bool LoadFromFile(const std::string& dataname, int t0, int t1);
  bool SaveToFile(const std::string& dataname, int t0, int t1) const;

  bool Valid() const {return _match.size()>0;}
 
public:
  int& operator()(int, int);
  int operator()(int, int) const;

  int t0() const {return _t0;} // timestep
  int t1() const {return _t1;}
  int n0() const {return _n0;}
  int n1() const {return _n1;}

private:
  std::string MatrixFileName(const std::string& dataname, int t0, int t1) const;

private:
  int _t0, _t1;
  int _n0, _n1;
  std::vector<int> _match; // match matrix
};

#endif
