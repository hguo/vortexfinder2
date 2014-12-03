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
#include "punctured_elem.h"
#include "vortex_object.h"

using namespace libMesh; 

/* 
 * \class   VortexExtractor
 * \author  Hanqi Guo
 * \brief   Vortex extractor for Condor2 output
*/
class VortexExtractor : public ParallelObject
{
public:
  VortexExtractor(const Parallel::Communicator &comm);
  ~VortexExtractor();

  void SetVerbose(int level=1);

  void SetMagneticField(const double B[3]);
  void SetKex(double Kex);
  void SetGaugeTransformation(bool); 

  void SetMesh(Mesh *);
  void SetSolution(AutoPtr<NumericVector<Number> > solution); 

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
  PuncturedElemMap<> _map; 
}; 

#endif
