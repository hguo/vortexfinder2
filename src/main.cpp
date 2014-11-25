#include <iostream>
#include <vector>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/dof_map.h>
#include "utils.h"

#define GAUGE 0

using namespace libMesh; 

int main(int argc, char **argv)
{
  // const char filename[] = "tslab.2.Bz0_1.exodus.short_out.e"; 
  const char filename[] = "tslab.3.Bz0_02.Nt1000.lu.512.e"; 
  const int timestep = 600;
  const double B[3] = {0.f, 0.f, 0.02f}; // magenetic field
  const double Kex = 0; 

  LibMeshInit init(argc, argv); 

  /// mesh 
  Mesh mesh(init.comm());
  ExodusII_IO exio(mesh);
  exio.read(filename);
  mesh.allow_renumbering(false); 
  mesh.prepare_for_use(); 

  /// equation systems
  EquationSystems eqsys(mesh); 

  NonlinearImplicitSystem &tsys = eqsys.add_system<NonlinearImplicitSystem>("GLsys"); 
  const unsigned int u_var = tsys.add_variable("u", FIRST, LAGRANGE);
  const unsigned int v_var = tsys.add_variable("v", FIRST, LAGRANGE); 

  eqsys.init(); 

  /// copy nodal data
  exio.copy_nodal_solution(tsys, "u", "u", timestep); 
  exio.copy_nodal_solution(tsys, "v", "v", timestep);

  /// testing
#if 1
  const DofMap &dof_map  = tsys.get_dof_map();  

  MeshBase::const_element_iterator it = mesh.active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = mesh.active_local_elements_end(); 

  std::vector<float> zeros; 

  for (; it!=end; it++) {
    const Elem *elem = *it;
   
#if 1
    for (int i=0; i<elem->n_sides(); i++) {
      AutoPtr<Elem> side = elem->side(i); 
      
      std::vector<dof_id_type> u_di, v_di; 
      dof_map.dof_indices(side.get(), u_di, u_var);
      dof_map.dof_indices(side.get(), v_di, v_var);
      
      double u[3] = {(*tsys.solution)(u_di[0]), (*tsys.solution)(u_di[1]), (*tsys.solution)(u_di[2])}, 
             v[3] = {(*tsys.solution)(v_di[0]), (*tsys.solution)(v_di[1]), (*tsys.solution)(v_di[2])};

      Node *nodes[3] = {side->get_node(0), side->get_node(1), side->get_node(2)};
      double x[3] = {nodes[0]->slice(0), nodes[1]->slice(0), nodes[2]->slice(0)}, 
             y[3] = {nodes[0]->slice(1), nodes[1]->slice(1), nodes[2]->slice(1)}, 
             z[3] = {nodes[0]->slice(2), nodes[1]->slice(2), nodes[2]->slice(2)};
      double X0[3] = {x[0], y[0], z[0]}, 
             X1[3] = {x[1], y[1], z[1]}, 
             X2[3] = {x[2], y[2], z[2]}; 
      double pos[3]; 

      double rho[3] = {sqrt(u[0]*u[0]+v[0]*v[0]), sqrt(u[1]*u[1]+v[1]*v[1]), sqrt(u[2]*u[2]+v[2]*v[2])}, 
             phi[3] = {atan2(v[0], u[0]), atan2(v[1], u[1]), atan2(v[2], u[2])}; 

      // check phase shift
      double flux = 0.f; // need to compute the flux correctly
#if GAUGE
      double delta[3] = {
        phi[1] - phi[0] - gauge_transformation(X0, X1, Kex, B), 
        phi[2] - phi[1] - gauge_transformation(X1, X2, Kex, B), 
        phi[0] - phi[2] - gauge_transformation(X2, X0, Kex, B)
      };
#else
      double delta[3] = {
        phi[1] - phi[0], 
        phi[2] - phi[1], 
        phi[0] - phi[2]
      };
#endif
      double delta1[3];  

      double sum = 0.f; 
      for (int k=0; k<3; k++) {
        delta1[k] = mod2pi(delta[k] + M_PI) - M_PI; 
        sum += delta1[k]; 
      }
      sum += flux; 
      double ps = sum / (2*M_PI); 
      if (abs(ps)<0.5f) continue;

#if GAUGE
      phi[1] = phi[0] + delta1[0]; 
      phi[2] = phi[1] + delta1[1];
      u[1] = rho[1] * cos(phi[1]);
      v[1] = rho[1] * sin(phi[1]); 
      u[2] = rho[2] * cos(phi[2]); 
      v[2] = rho[2] * sin(phi[2]);
#endif

      bool succ = find_zero_triangle(u, v, x, y, z, pos); 
      if (succ) {
        zeros.push_back(pos[0]); 
        zeros.push_back(pos[1]); 
        zeros.push_back(pos[2]);
#if 0
        fprintf(stderr, "elem=%p, side=%d, u={%f, %f, %f}, v={%f, %f, %f}, rho={%f, %f, %f}, phi={%f, %f, %f}, ps=%f, pos={%f, %f, %f}\n", 
            elem, i, 
            u[0], u[1], u[2], v[0], v[1], v[2], 
            rho[0], rho[1], rho[2], phi[0], phi[1], phi[2],  
            ps, pos[0], pos[1], pos[2]);
#endif
      } else {
        fprintf(stderr, "punctured but zero not found\n"); 
      }
    }
#endif
  }

  int npts = zeros.size()/3; 
  fprintf(stderr, "total points: %d\n", npts); 
  FILE *fp = fopen("out", "wb");
  fwrite(&npts, sizeof(int), 1, fp); 
  fwrite(zeros.data(), sizeof(float), zeros.size(), fp); 
  fclose(fp); 
#endif

  return 0; 
}
