#ifndef _CONDOR2EXTRACTOR_H
#define _CONDOR2EXTRACTOR_H

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
#include "Extractor.h"
#include "PuncturedElem.h"
#include "io/Condor2Dataset.h"
#include "vortex/VortexObject.h"

using namespace libMesh; 

/* 
 * \class   Condor2VortexExtractor
 * \author  Hanqi Guo
 * \brief   Vortex extractor for Condor2 output
*/
class Condor2VortexExtractor : public ParallelObject, public VortexExtractor
{
public:
  Condor2VortexExtractor(const Parallel::Communicator &comm);
  ~Condor2VortexExtractor();

  void SetVerbose(int level=1);

  void SetMagneticField(const double B[3]);
  void SetKex(double Kex);
  void SetGaugeTransformation(bool); 

  /// future features...
  // void SetMesh(Mesh *);
  // void SetSolution(AutoPtr<NumericVector<Number> > solution); 

  void SetDataset(const GLDataset* ds);
  const Condor2Dataset* Dataset() const {return _ds;}

  // to be deprecated 
  void LoadData(const std::string& filename); 
  void LoadTimestep(int timestep);

  void Extract();
  void Trace(); 
  void WriteVortexObjects(const std::string& filename); 

protected:
  bool Verbose(int level=1) {return level <= _verbose;} 

private: 
  int _verbose; 
 
  const Condor2Dataset *_ds;

  // to be deprecated
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
  PuncturedElemMap<> _punctured_elems; 
  std::vector<VortexObject> _vortex_objects; 
}; 

#endif
