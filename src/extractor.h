#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include <string>
#include <map>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/dof_map.h>
#include "vortex.h"

using namespace libMesh; 

// a stand alone vortex extractor
class VortexExtractor : public ParallelObject
{
public:
  VortexExtractor(const Parallel::Communicator &comm);
  ~VortexExtractor();

  void SetVerbose(int level=1);

  void SetMagneticField(const double B[3]);
  void SetKex(double Kex);
  void SetGaugeTransformation(bool); 

  void LoadData(const std::string& filename); 
  void LoadTimestep(int timestep); 

  void Extract();
  void Trace(); 

protected:
  bool Verbose(int level=1) {return level <= _verbose;} 

private: 
  int _verbose; 
  
  int _timestep; 
  double _B[3]; // magenetic field
  double _Kex; // Kex
  bool _gauge; 

  UnstructuredMesh *_mesh;
  ExodusII_IO *_exio; 
  EquationSystems *_eqsys;
  NonlinearImplicitSystem *_tsys;
  unsigned int _u_var, _v_var;

private:
  template <typename T>
  struct PuncturedElem {
    const Elem *elem; 
    std::bitset<8> bits; 
    std::vector<T> pos;

    PuncturedElem() : pos(12) {} // for easier access.. should later reduce memory footprint 
    bool Valid() const {return bits.any();} 
    int Chirality(int face) const {
      if (!bits[face]) return 0; // face not punctured
      else return bits[face+4] ? 1 : -1; 
    }
    void SetChirality(int face, int chirality) {if (chirality==1) bits[face+4] = 1;}
    bool IsPunctured(int face) {return bits[face];}
    void SetPuncturedFace(int face) {bits[face] = 1;}
    void SetPuncturedPoint(int face, const T* p) {pos[face*3] = p[0]; pos[face*3+1] = p[1]; pos[face*3+2] = p[2];}
    void GetPuncturedPoint(int face, T* p) {p[0] = pos[face*3]; p[1] = pos[face*3+1]; p[2] = pos[face*3+2];}
  }; 

  std::map<const Elem*, PuncturedElem<double> > _punctured_elems; 
  typedef std::map<const Elem*, PuncturedElem<double> >::const_iterator punctured_elem_iterator;
  std::list<punctured_elem_iterator> _traced_punctured_elems; 

private:
  void Trace(VortexObject& vortex, punctured_elem_iterator iterator, int direction); 
}; 

#endif
