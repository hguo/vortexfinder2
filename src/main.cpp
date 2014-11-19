#include <iostream>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/dof_map.h>

using namespace libMesh; 

int main(int argc, char **argv)
{
  // const char filename[] = "tslab.2.Bz0_1.exodus.short_out.e"; 
  const char filename[] = "tslab.3.Bz0_02.Nt1000.lu.512.e"; 

  LibMeshInit init(argc, argv); 

  /// mesh 
  Mesh mesh(init.comm());
  ExodusII_IO exio(mesh);
  exio.read(filename);
  mesh.prepare_for_use(); 
  // mesh.read(filename); 
  mesh.print_info(); 

  /// equation systems
  EquationSystems eqsys(mesh); 

  NonlinearImplicitSystem &tsys = eqsys.add_system<NonlinearImplicitSystem>("GLsys"); 
  tsys.add_variable("u", FIRST, LAGRANGE);
  tsys.add_variable("v", FIRST, LAGRANGE); 

  System &asys = eqsys.add_system<System>("Auxsys"); 
  asys.add_variable("Ax", FIRST, LAGRANGE); 
  asys.add_variable("Ay", FIRST, LAGRANGE); 
  asys.add_variable("Az", FIRST, LAGRANGE); 
  asys.add_variable("rho", FIRST, LAGRANGE); 
  asys.add_variable("phi", FIRST, LAGRANGE); 

  eqsys.init(); 
  eqsys.print_info(); 


  /// copy nodal data
  // tsys.solution->zero_clone();  
  // asys.solution->zero_clone();  

  exio.copy_nodal_solution(tsys, "u", "u", 600); 
  exio.copy_nodal_solution(tsys, "v", "v", 600);
#if 0
  exio.copy_nodal_solution(asys, "Ax", "A_x", 1); 
  exio.copy_nodal_solution(asys, "Ay", "A_y", 1); 
  exio.copy_nodal_solution(asys, "Az", "A_z", 1); 
  exio.copy_nodal_solution(asys, "rho", "rho", 1); 
  exio.copy_nodal_solution(asys, "phi", "phi", 1);
#endif

  /// testing
#if 0
  const DofMap &dof_map = tsys.get_dof_map(); 

  MeshBase::const_element_iterator it = mesh.active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = mesh.active_local_elements_end(); 

  for (; it!=end; it++) {
    const Elem *elem = *it;
    std::vector<dof_id_type> di; 
    dof_map.dof_indices(elem, di);
  
    // std::set<const Elem*> neighbors; 
    // elem->find_point_neighbors(neighbors);
    // elem->find_edge_neighbors(neighbors);

    // for (int i=0; i<elem->n_neighbors(); i++) 
    //   fprintf(stderr, "element %p, neighbor=%p\n", elem, elem->neighbor(i)); 

#if 1
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
#endif

  return 0; 
}
