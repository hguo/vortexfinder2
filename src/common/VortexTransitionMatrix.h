#ifndef _VORTEX_TRANSITION_MATRIX_H
#define _VORTEX_TRANSITION_MATRIX_H

#include <string>
#include <vector>
#include <map>
#include <set>

enum {
  VORTEX_EVENT_DUMMY = 0,
  VORTEX_EVENT_BIRTH = 1,
  VORTEX_EVENT_DEATH = 2,
  VORTEX_EVENT_MERGE = 3,
  VORTEX_EVENT_SPLIT = 4,
  VORTEX_EVENT_RECOMBINATION = 5, 
  VORTEX_EVENT_COMPOUND = 6
};

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

  bool Valid() const {return _match.size()>0;}
  
public: // modulars
  void Modularize();
  int NModules() const {return _lhss.size();}
  void GetModule(int i, std::vector<int>& lhs, std::vector<int>& rhs, int &event) const;
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
  std::vector<std::vector<int> > _lhss, _rhss;
  std::vector<int> _events;
};

#endif
