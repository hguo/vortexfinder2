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

using namespace libMesh; 

static inline bool find_zero_triangle(float r[3], float i[3], float lambda[3])
{
  float D = r[0]*i[1] + r[1]*i[2] + r[2]*i[0] - r[2]*i[1] - r[1]*i[0] - r[0]*i[2]; // TODO: check if D=0?
  float det[3] = {
    r[1]*i[2] - r[2]*i[1], 
    r[2]*i[0] - r[0]*i[2], 
    r[0]*i[1] - r[1]*i[0]
  };

  lambda[0] = det[0]/D; 
  lambda[1] = det[1]/D; 
  lambda[2] = det[2]/D; 
  
  if (lambda[0]>=0 && lambda[1]>=0 && lambda[2]>=0) return true; 
  else return false; 
}

static inline bool find_zero_triangle(float r[3], float i[3], float x[3], float y[3], float z[3], float pos[3])
{
  float lambda[3]; 
  if (!find_zero_triangle(r, i, lambda)) return false; 

  float T[3][2] = {{x[0]-x[2], x[1]-x[2]}, 
                   {y[0]-y[2], y[1]-y[2]}, 
                   {z[0]-z[2], z[1]-z[2]}}; 

  pos[0] = T[0][0]*lambda[0] + T[0][1]*lambda[1] + x[2]; 
  pos[1] = T[1][0]*lambda[0] + T[1][1]*lambda[1] + y[2]; 
  pos[2] = T[2][0]*lambda[0] + T[2][1]*lambda[1] + z[2]; 

  return true; 
}

int main(int argc, char **argv)
{
  // const char filename[] = "tslab.2.Bz0_1.exodus.short_out.e"; 
  const char filename[] = "tslab.3.Bz0_02.Nt1000.lu.512.e"; 
  const int timestep = 600; 

  LibMeshInit init(argc, argv); 

  /// mesh 
  Mesh mesh(init.comm());
  ExodusII_IO exio(mesh);
  exio.read(filename);
  mesh.prepare_for_use(); 
  // mesh.read(filename); 
  // mesh.print_info(); 

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
  // eqsys.print_info(); 

  /// copy nodal data
  // tsys.solution->zero_clone();  
  // asys.solution->zero_clone();  
  exio.copy_nodal_solution(tsys, "u", "u", timestep); 
  exio.copy_nodal_solution(tsys, "v", "v", timestep);
  exio.copy_nodal_solution(asys, "Ax", "A_x", timestep); 
  exio.copy_nodal_solution(asys, "Ay", "A_y", timestep); 
  exio.copy_nodal_solution(asys, "Az", "A_z", timestep); 
  exio.copy_nodal_solution(asys, "rho", "rho", timestep); 
  exio.copy_nodal_solution(asys, "phi", "phi", timestep);

  /// testing
#if 1
  const DofMap &dof_map = tsys.get_dof_map(); 

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

      Node *nodes[3] = {side->get_node(0), side->get_node(1), side->get_node(2)}; 
      float x[3] = {nodes[0]->slice(0), nodes[1]->slice(0), nodes[2]->slice(0)}, 
            y[3] = {nodes[0]->slice(1), nodes[1]->slice(1), nodes[2]->slice(1)}, 
            z[3] = {nodes[0]->slice(2), nodes[1]->slice(2), nodes[2]->slice(2)}; 
      float u[3] = {(*tsys.solution)(di[0]), (*tsys.solution)(di[1]), (*tsys.solution)(di[2])}, 
            v[3] = {(*tsys.solution)(di[3]), (*tsys.solution)(di[4]), (*tsys.solution)(di[5])};
      float pos[3]; 

      bool succ = find_zero_triangle(u, v, x, y, z, pos); 
      if (succ) {
        zeros.push_back(pos[0]); 
        zeros.push_back(pos[1]); 
        zeros.push_back(pos[2]); 
        // fprintf(stderr, "elem=%p, side=%d, u={%f, %f, %f}, v={%f, %f, %f}, pos={%f, %f, %f}\n", 
        //     elem, i, u[0], u[1], u[2], v[0], v[1], v[2], pos[0], pos[1], pos[2]); 
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
