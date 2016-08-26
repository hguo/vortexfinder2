#ifndef _VORTEX_TRANSITION_MATRIX_H
#define _VORTEX_TRANSITION_MATRIX_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include "common/VortexEvents.h"

class VortexTransitionMatrix {
public:
  VortexTransitionMatrix();
  VortexTransitionMatrix(int t0, int t1, int n0, int n1);
  ~VortexTransitionMatrix();

public: // IO
  bool LoadFromFile(const std::string& filename);
  bool SaveToFile(const std::string& filename) const;

  bool LoadFromFile(const std::string& dataname, int t0, int t1);
  bool SaveToFile(const std::string& dataname, int t0, int t1) const;

  void SetToDummy() {_n0 = _n1 = 0; _match.push_back(0);}

  bool Valid() const {return _match.size()>0;}

  void Print() const;
  
public: // modulars
  void Modularize();
  int NModules() const {return _lhss.size();}
  void GetModule(int i, std::set<int>& lhs, std::set<int>& rhs, int &event) const;
  void Normalize();
 
public: // access
  int& operator()(int, int);
  int operator()(int, int) const;
  int& at(int i, int j);
  int at(int i, int j) const;

  int t0() const {return _t0;} // timestep
  int t1() const {return _t1;}
  int n0() const {return _n0;}
  int n1() const {return _n1;}

  int colsum(int j) const;
  int rowsum(int i) const;

private:
  std::string MatrixFileName(const std::string& dataname, int t0, int t1) const;

private:
  int _t0, _t1;
  int _n0, _n1;
  std::vector<int> _match; // match matrix

  // modulars
  std::vector<std::set<int> > _lhss, _rhss;
  std::vector<int> _events;
};

#endif
