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

  NonlinearImplicitSystem &tsys = eqsys.add_system<NonlinearImplicitSystem>("GLsys"); 
  tsys.add_variable("u", FIRST, LAGRANGE);
  tsys.add_variable("v", FIRST, LAGRANGE); 

  System &asys = eqsys.add_system<System>("Auxsys");
#if 0
  asys.add_variable("Ax", FIRST, LAGRANGE); 
  asys.add_variable("Ay", FIRST, LAGRANGE); 
  asys.add_variable("Az", FIRST, LAGRANGE); 
  asys.add_variable("rho", FIRST, LAGRANGE); 
  asys.add_variable("phi", FIRST, LAGRANGE);
#endif

  eqsys.init(); 
  // eqsys.print_info(); 

  /// copy nodal data
  tsys.solution->zero_clone();  
  // asys.solution->zero_clone();  
  exio.copy_nodal_solution(tsys, "u", "u", timestep); 
  exio.copy_nodal_solution(tsys, "v", "v", timestep);
#if 0
  exio.copy_nodal_solution(asys, "Ax", "A_x", timestep); 
  exio.copy_nodal_solution(asys, "Ay", "A_y", timestep); 
  exio.copy_nodal_solution(asys, "Az", "A_z", timestep); 
  exio.copy_nodal_solution(asys, "rho", "rho", timestep); 
  exio.copy_nodal_solution(asys, "phi", "phi", timestep);
#endif

  /// testing
#if 1
  const DofMap &dof_map  = tsys.get_dof_map(), 
               &dof_map1 = asys.get_dof_map(); 

  MeshBase::const_element_iterator it = mesh.active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = mesh.active_local_elements_end(); 

  std::vector<float> zeros; 

  for (; it!=end; it++) {
    const Elem *elem = *it;
   
#if 1
    for (int i=0; i<elem->n_sides(); i++) {
      AutoPtr<Elem> side = elem->side(i); 
      
      std::vector<dof_id_type> di; 
      dof_map.dof_indices(side.get(), di);
      
      std::vector<dof_id_type> di1; // aux
      dof_map1.dof_indices(side.get(), di1);

      Node *nodes[3] = {side->get_node(0), side->get_node(1), side->get_node(2)};
      float x[3] = {nodes[0]->slice(0), nodes[1]->slice(0), nodes[2]->slice(0)}, 
            y[3] = {nodes[0]->slice(1), nodes[1]->slice(1), nodes[2]->slice(1)}, 
            z[3] = {nodes[0]->slice(2), nodes[1]->slice(2), nodes[2]->slice(2)};
      float X0[3] = {x[0], y[0], z[0]}, 
            X1[3] = {x[1], y[1], z[1]}, 
            X2[3] = {x[2], y[2], z[2]}; 
     
#if 0
      float Ax[3] = {(*asys.solution)(di1[0]), (*asys.solution)(di1[1]), (*asys.solution)(di1[2])}; 
      float Ay[3] = {(*asys.solution)(di1[3]), (*asys.solution)(di1[4]), (*asys.solution)(di1[5])}; 
      float Az[3] = {(*asys.solution)(di1[6]), (*asys.solution)(di1[7]), (*asys.solution)(di1[8])}; 
      float rho[3]= {(*asys.solution)(di1[9]), (*asys.solution)(di1[10]), (*asys.solution)(di1[11])}; 
      float phi[3]= {(*asys.solution)(di1[12]), (*asys.solution)(di1[13]), (*asys.solution)(di1[14])};
#endif

      float u[3] = {(*tsys.solution)(di[0]), (*tsys.solution)(di[1]), (*tsys.solution)(di[2])}, 
            v[3] = {(*tsys.solution)(di[3]), (*tsys.solution)(di[4]), (*tsys.solution)(di[5])};
      float pos[3]; 

      float rho[3] = {sqrt(u[0]*u[0]+v[0]*v[0]), sqrt(u[1]*u[1]+v[1]*v[1]), sqrt(u[2]*u[2]+v[2]*v[2])}, 
            phi[3] = {atan2(v[0], u[0]), atan2(v[1], u[1]), atan2(v[2], u[2])}; 

#if 0
      fprintf(stderr, "phi={%f, %f, %f}, phi1={%f, %f, %f}\n", 
          phi[0], phi[1], phi[2], 
          atan2(v[0], u[0]), atan2(v[1], u[1]), atan2(v[2], u[2]));
#endif

      // check phase shift
      float flux = 0.f; // need to compute the flux correctly
#if GAUGE
      float delta[3] = {
        phi[1] - phi[0] + gauge_transformation(X0, X1, Kex, B), 
        phi[2] - phi[1] + gauge_transformation(X1, X2, Kex, B), 
        phi[0] - phi[2] + gauge_transformation(X2, X0, Kex, B)
      };
#else
      float delta[3] = {
        phi[1] - phi[0], 
        phi[2] - phi[1], 
        phi[0] - phi[2]
      };
#endif
      float delta1[3];  

      float sum = 0.f; 
      for (int k=0; k<3; k++) {
        delta1[k] = mod2pi(delta[k] + M_PI) - M_PI; 
        sum += delta1[k]; 
      }
      sum += flux; 
      float ps = sum / (2*M_PI); 
      if (fabs(ps)<0.9f) continue;

#if GAUGE
      phi[1] = phi[0] + delta1[0]; 
      phi[2] = phi[1] + delta1[2];
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
      }
    }
#endif

#if 0
    std::vector<dof_id_type> di; 
    dof_map.dof_indices(elem, di);
    for (int i=0; i<di.size(); i++) 
      fprintf(stderr, "element %p, i=%d, di.size=%lu, dof_index=%u, value=%f\n", 
          elem, i, di.size(), di[i], (*tsys.solution)(di[i]));  
#endif
#if 0
    for (unsigned int i=0; i<elem->n_nodes(); i++) {
      const Node *node = elem->get_node(i);
      unsigned int id = elem->node(i); 
      fprintf(stderr, "element %p, id=%d, coords={%f, %f, %f}\n", elem, id, (*node)(0), (*node)(1), (*node)(2));
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
