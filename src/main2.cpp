#include <iostream>
#include <vector>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/vtk_io.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/dof_map.h>
#include "utils.h"

using namespace libMesh; 

int main(int argc, char **argv)
{
  // const char filename[] = "tslab.2.Bz0_1.exodus.short_out.e"; 
  const char filename[] = "tslab.3.Bz0_02.Nt1000.lu.512.e"; 
  const int timestep = 13;
  const float B[3] = {0.f, 0.f, 0.02f}; // magenetic field
  const float Kex = 0; 

  LibMeshInit init(argc, argv); 

  /// mesh 
  Mesh mesh(init.comm());
  ExodusII_IO exio(mesh);
  exio.read(filename);
  mesh.allow_renumbering(false); 
  mesh.prepare_for_use();
  // mesh.read(filename); 
  // mesh.print_info(); 

  /// equation systems
  EquationSystems eqsys(mesh); 

#if 1
  NonlinearImplicitSystem &tsys = eqsys.add_system<NonlinearImplicitSystem>("GLsys"); 
  const unsigned int u_var = tsys.add_variable("u", FIRST, LAGRANGE);
  const unsigned int v_var = tsys.add_variable("v", FIRST, LAGRANGE);
#endif
  
  System &asys = eqsys.add_system<System>("Auxsys");
  const unsigned int rho_var = asys.add_variable("rho", FIRST, LAGRANGE);

  eqsys.init(); 
  // eqsys.print_info(); 

  /// copy nodal data
  // exio.copy_nodal_solution(tsys, "u", "u", timestep); 
  // exio.copy_nodal_solution(tsys, "v", "v", timestep);
  exio.copy_nodal_solution(asys, "rho", "rho", timestep); 

  /// VTK output
#if 0
  VTKIO vtkio(mesh);
  std::vector<std::string> varnames; 
  varnames.push_back("rho"); 
  
  std::vector<Number> solution; 
  asys.solution->localize(solution); 
  vtkio.write_nodal_data("out.vtk", solution, varnames);
#endif

  /// another exodus output
#if 0
  ExodusII_IO exout(mesh);
  std::vector<std::string> varnames; 
  varnames.push_back("rho");

  std::vector<Number> solution(asys.solution->size());
  for (int i=0; i<solution.size(); i++) 
    solution[i] = (*asys.solution)(i); 
  // asys.solution->localize(solution); 
  exout.write_nodal_data("out.e", solution, varnames); 
#endif

  // return 1; 

#if 1
  /// testing
  const DofMap &dof_map  = tsys.get_dof_map(); 

  MeshBase::const_element_iterator it = mesh.active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = mesh.active_local_elements_end(); 

  std::vector<float> vertices, rhos;  

  for (; it!=end; it++) {
    Elem *elem = *it; 
    
    std::vector<dof_id_type> u_di, v_di; 
    dof_map.dof_indices(elem, u_di, u_var);
    dof_map.dof_indices(elem, v_di, v_var);

    std::vector<Number> u_val, v_val; 
    tsys.solution->get(u_di, u_val); 
    tsys.solution->get(u_di, v_val); 

    for (int i=0; i<elem->n_nodes(); i++) {
      double X[3] = {elem->point(i).slice(0), elem->point(i).slice(1), elem->point(i).slice(2)};
      for (int k=0; k<3; k++) 
        vertices.push_back(X[k]);

      float u = u_val[i], v = v_val[i]; 
      float rho = sqrt(u*u + v*v);  

      // double u = (*tsys.solution)(u_di[i]),  
      //       v = (*tsys.solution)(v_di[i]);
      // double rho = sqrt(u*u+v*v);

      //fprintf(stderr, "pos={%f, %f, %f}, u_di=%d, v_di=%d, u=%f, v=%f, rho=%f\n", 
      //    X[0], X[1], X[2], u_di[i], v_di[i], u, v, rho); 

      rhos.push_back(rho); 
    }
  }
  
  int npts = vertices.size()/3; 
  fprintf(stderr, "total points: %d\n", npts); 
  FILE *fp = fopen("out", "wb");
  fwrite(&npts, sizeof(int), 1, fp); 
  fwrite(vertices.data(), sizeof(float), vertices.size(), fp);
  fwrite(rhos.data(), sizeof(float), rhos.size(), fp); 
  fclose(fp); 
#endif
  return 0; 
}
