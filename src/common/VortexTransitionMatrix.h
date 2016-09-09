#ifndef _VORTEX_TRANSITION_MATRIX_H
#define _VORTEX_TRANSITION_MATRIX_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include "def.h"
#include "common/diy-ext.hpp"
#include "common/VortexEvents.h"
#include "common/Interval.h"

class VortexTransitionMatrix {
  friend class diy::Serialization<VortexTransitionMatrix>;
public:
  VortexTransitionMatrix();
  VortexTransitionMatrix(int t0, int t1, int n0, int n1);
  VortexTransitionMatrix(Interval, int n0, int n1);
  ~VortexTransitionMatrix();

public: // IO
  void SetToDummy() {_n0 = _n1 = 0; _match.clear();}
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

  int t0() const {return _interval.first;} // timestep
  int t1() const {return _interval.second;}
  int n0() const {return _n0;}
  int n1() const {return _n1;}

  Interval GetInterval() const {return _interval;}
  void SetInterval(const Interval &i) {_interval = i;}

  int colsum(int j) const;
  int rowsum(int i) const;

private:
  // std::string MatrixFileName(const std::string& dataname, int t0, int t1) const;

private:
  Interval _interval;
  int _n0, _n1;
  std::vector<int> _match; // match matrix

  // modulars
  std::vector<std::set<int> > _lhss, _rhss;
  std::vector<int> _events;
};


///////////
template <> struct diy::Serialization<VortexTransitionMatrix> {
  static void save(diy::BinaryBuffer& bb, const VortexTransitionMatrix& m) {
    diy::save(bb, m._interval);
    diy::save(bb, m._n0);
    diy::save(bb, m._n1);
    diy::save(bb, m._match);
  }

  static void load(diy::BinaryBuffer&bb, VortexTransitionMatrix& m) {
    diy::load(bb, m._interval);
    diy::load(bb, m._n0);
    diy::load(bb, m._n1);
    diy::load(bb, m._match);
    if (m.Valid()) m.Normalize();
  }
};

#endif
