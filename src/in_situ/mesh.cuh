#ifndef _MESH_CUH
#define _MESH_CUH

__device__ __host__
inline void nid2nidx3D(const vf2gpu_hdr_t& h, unsigned int id, int idx[3])
{
  const int s = h.d[0] * h.d[1]; 
  const int k = id / s; 
  const int j = (id - k*s) / h.d[0]; 
  const int i = id - k*s - j*h.d[0]; 

  idx[0] = i; idx[1] = j; idx[2] = k;
}

__device__ __host__
inline void nid2nidx2D(const vf2gpu_hdr_t& h, unsigned int id, int idx[2])
{
  int j = id / h.d[0]; 
  int i = id - j*h.d[0];

  idx[0] = i; idx[1] = j;
}

__device__ __host__
inline unsigned int nidx2nid3D(const vf2gpu_hdr_t& h, const int idx_[3])
{
  int idx[3] = {idx_[0], idx_[1], idx_[2]};
  for (int i=0; i<3; i++) {
    idx[i] = idx[i] % h.d[i];
    if (idx[i] < 0)
      idx[i] += h.d[i];
  }
  return idx[0] + h.d[0] * (idx[1] + h.d[1] * idx[2]); 
}

__device__ __host__
inline unsigned int nidx2nid2D(const vf2gpu_hdr_t& h, const int idx_[2])
{
  int idx[2] = {idx_[0], idx_[1]};
  for (int i=0; i<2; i++) {
    idx[i] = idx[i] % h.d[i];
    if (idx[i] < 0)
      idx[i] += h.d[i];
  }
  return idx[0] + h.d[0] * idx[1];
}

__device__ __host__
inline bool valid_nidx3D(const vf2gpu_hdr_t& h, const int idx[3])
{
  bool v[3] = {
    idx[0]>=0 && idx[0]<h.d[0],
    idx[1]>=0 && idx[1]<h.d[1],
    idx[2]>=0 && idx[2]<h.d[2]
  };
  return v[0] && v[1] && v[2];
}

__device__ __host__
inline bool valid_nidx2D(const vf2gpu_hdr_t& h, const int idx[2])
{
  bool v[2] = {
    idx[0]>=0 && idx[0]<h.d[0],
    idx[1]>=0 && idx[1]<h.d[1]
  };
  return v[0] && v[1];
}

__device__ __host__ 
inline bool valid_cidx3D_hex(const vf2gpu_hdr_t& h, const int cidx[3])
{
  bool v[3] = {
    cidx[0]>=0 && (cidx[0]<h.d[0] - (!h.pbc[0])),
    cidx[1]>=0 && (cidx[1]<h.d[1] - (!h.pbc[1])),
    cidx[2]>=0 && (cidx[2]<h.d[2] - (!h.pbc[2]))
  };
  return v[0] && v[1] && v[2];
}

__device__ __host__ 
inline bool valid_cidx2D(const vf2gpu_hdr_t& h, const int cidx[2])
{
  bool v[2] = {
    cidx[0]>=0 && (cidx[0]<h.d[0] - (!h.pbc[0])),
    cidx[1]>=0 && (cidx[1]<h.d[1] - (!h.pbc[1]))
  };
  return v[0] && v[1];
}

__device__ __host__ 
inline unsigned int cidx2cid3D_hex(const vf2gpu_hdr_t& h, const int cidx[3])
{
  return nidx2nid3D(h, cidx);
}

__device__ __host__ 
inline unsigned int cidx2cid2D_hex(const vf2gpu_hdr_t& h, const int cidx[2])
{
  return nidx2nid2D(h, cidx);
}

__device__ __host__
inline unsigned int fidx2fid3D_hex(const vf2gpu_hdr_t& h, const int fidx[4])
{
  return nidx2nid3D(h, fidx)*3 + fidx[3];
}

__device__ __host__
inline void fid2fidx3D_tet(const vf2gpu_hdr_t& h, unsigned int id, int idx[4])
{
  unsigned int nid = id / 12;
  nid2nidx3D(h, nid, idx);
  idx[3] = id % 12;
}

__device__ __host__
inline unsigned int fidx2fid2D(const vf2gpu_hdr_t& h, const int fidx[3])
{
  return nidx2nid2D(h, fidx);
}

__device__ __host__
inline void fid2fidx3D_hex(const vf2gpu_hdr_t& h, unsigned int id, int idx[4])
{
  unsigned int nid = id / 3;
  nid2nidx3D(h, nid, idx);
  idx[3] = id % 3;
}

__device__ __host__
bool valid_fidx3D_tet(const vf2gpu_hdr_t& h, const int fidx[4])
{
  if (fidx[3]<0 || fidx[3]>=12) return false;
  else {
    int o[3] = {0};
    for (int i=0; i<3; i++)
      if (h.pbc[i]) {
        if (fidx[i] < 0 || fidx[i] >= h.d[i]) return false;
      } else {
        if (fidx[i] < 0 || fidx[i] > h.d[i]-1) return false;
        else if (fidx[i] == h.d[i]-1) o[i] = 1;
      }
    
    const int sum = o[0] + o[1] + o[2];
    if (sum == 0) return true;
    else if (o[0] + o[1] + o[2] > 1) return false;
    else if (o[0] && (fidx[3] == 4 || fidx[3] == 5)) return true;
    else if (o[1] && (fidx[3] == 2 || fidx[3] == 3)) return true; 
    else if (o[2] && (fidx[3] == 0 || fidx[3] == 1)) return true;
    else return false;
  }
}

__device__ __host__
inline void fid2fidx2D(const vf2gpu_hdr_t& h, unsigned int id, int idx[2])
{
  // unsigned int nid = id;
  nid2nidx2D(h, id, idx);
}

__device__ __host__
bool valid_fidx2D(const vf2gpu_hdr_t& h, const int fidx[3])
{
  return valid_cidx2D(h, fidx);
}

__device__ __host__
bool valid_fidx3D_hex(const vf2gpu_hdr_t& h, const int fidx[4])
{
  if (fidx[3]<0 || fidx[3]>=3) return false;
  else {
    int o[3] = {0}; 
    for (int i=0; i<3; i++) 
      if (h.pbc[i]) {
        if (fidx[i]<0 || fidx[i]>=h.d[i]) return false;
      } else {
        if (fidx[i]<0 || fidx[i]>h.d[i]-1) return false;
        else if (fidx[i] == h.d[i]-1) o[i] = 1;
      }

    const int sum = o[0] + o[1] + o[2];
    if (sum == 0) return true;
    else if (o[0] + o[1] + o[2] > 1) return false;
    else if (o[0] && fidx[3] == 0) return true; 
    else if (o[1] && fidx[3] == 1) return true;
    else if (o[2] && fidx[3] == 2) return true;
    else return false;
  }
}

__device__ __host__
inline void eid2eidx3D_tet(const vf2gpu_hdr_t& h, unsigned int id, int idx[4])
{
  unsigned int nid = id / 7;
  nid2nidx3D(h, nid, idx);
  idx[3] = id % 7;
}

__device__ __host__
inline void eid2eidx3D_hex(const vf2gpu_hdr_t& h, unsigned int id, int idx[4]) 
{
  unsigned int nid = id / 3;
  nid2nidx3D(h, nid, idx);
  idx[3] = id % 3;
}

__device__ __host__
void eid2eidx2D(const vf2gpu_hdr_t& h, unsigned int id, int idx[3])
{
  const unsigned int nid = id / 2;
  nid2nidx2D(h, nid, idx);
  idx[2] = id % 2;
}

__device__ __host__
inline bool valid_eidx3D_tet(const vf2gpu_hdr_t& h, const int eidx[4])
{
  if (eidx[3]<0 || eidx[3]>=7) return false;
  else {
    for (int i=0; i<3; i++)
      if (h.pbc[i]) {
        if (eidx[i] < 0 || eidx[i] >= h.d[i]) return false;
      } else {
        if (eidx[i] < 0 || eidx[i] >= h.d[i]-1) return false;
      }
    return true;
  }
}

__device__ __host__
inline bool valid_eidx3D_hex(const vf2gpu_hdr_t& h, const int eidx[4])
{
  if (eidx[3]<0 || eidx[3]>=3) return false;
  else {
    for (int i=0; i<3; i++) 
      if (h.pbc[i]) {
        if (eidx[i]<0 || eidx[i]>=h.d[i]) return false;
      } else {
        if (eidx[i]<0 || eidx[i]>=h.d[i]-1) return false;
      }
    return true;
  }
}

__device__ __host__
inline bool valid_eidx2D(const vf2gpu_hdr_t& h, const int eidx[3])
{
  if (eidx[2]<0 || eidx[2]>=2) return false;
  else return valid_cidx2D(h, eidx);
}

__device__ __host__
inline bool fid2nodes3D_tet(const vf2gpu_hdr_t& h, unsigned int id, int nidxs[3][3])
{
  const int nodes_idx[12][3][3] = { // 12 types of faces
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}}, // ABC
    {{0, 0, 0}, {1, 1, 0}, {0, 1, 0}}, // ACD
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}}, // ABF
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}}, // AEF
    {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}}, // ADE
    {{0, 1, 0}, {0, 0, 1}, {0, 1, 1}}, // DEH
    {{0, 0, 0}, {0, 1, 0}, {1, 0, 1}}, // ADF
    {{0, 1, 0}, {1, 0, 1}, {1, 1, 1}}, // DFG
    {{0, 1, 0}, {0, 0, 1}, {1, 0, 1}}, // DEF
    {{1, 1, 0}, {0, 1, 0}, {1, 0, 1}}, // CDF
    {{0, 0, 0}, {1, 1, 0}, {1, 0, 1}}, // ACF
    {{0, 1, 0}, {0, 0, 1}, {1, 1, 1}}  // DEG
  };

  int fidx[4];
  fid2fidx3D_tet(h, id, fidx);

  if (valid_fidx3D_tet(h, fidx)) {
    const int type = fidx[3];
    for (int p=0; p<3; p++) 
      for (int q=0; q<3; q++) 
        nidxs[p][q] = fidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}

__device__ __host__
inline bool fid2nodes3D_hex(const vf2gpu_hdr_t &h, unsigned int id, int nidxs[4][3])
{
  const int nodes_idx[3][4][3] = { // 3 types of faces
    {{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}, // YZ
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}}, // ZX
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}  // XY
  };
  
  int fidx[4];
  fid2fidx3D_hex(h, id, fidx);

  if (valid_fidx3D_hex(h, fidx)) {
    const int type = fidx[3];
    for (int p=0; p<4; p++) 
      for (int q=0; q<3; q++) 
        nidxs[p][q] = fidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}

__device__ __host__
inline bool fid2nodes2D(const vf2gpu_hdr_t& h, unsigned int id, int nidxs[4][3])
{
  const int nodes_idx[4][3] = { // only 1 type of faces
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};

  int fidx[2];
  fid2fidx2D(h, id, fidx);

  if (valid_fidx2D(h, fidx)) {
    for (int p=0; p<4; p++) 
      for (int q=0; q<2; q++)
        nidxs[p][q] = fidx[q] + nodes_idx[p][q];
    return true;
  }
  else 
    return false;
}

__device__ __host__
inline bool eid2nodes3D_tet(const vf2gpu_hdr_t& h, unsigned int eid, int nidxs[2][3])
{
  const int nodes_idx[7][2][3] = { // 7 types of edges
    {{0, 0, 0}, {1, 0, 0}}, // AB
    {{0, 0, 0}, {1, 1, 0}}, // AC
    {{0, 0, 0}, {0, 1, 0}}, // AD
    {{0, 0, 0}, {0, 0, 1}}, // AE
    {{0, 0, 0}, {1, 0, 1}}, // AF
    {{0, 1, 0}, {0, 0, 1}}, // DE
    {{0, 1, 0}, {1, 0, 1}}, // DF
  };
  
  int eidx[4];
  eid2eidx3D_tet(h, eid, eidx);

  if (valid_eidx3D_tet(h, eidx)) {
    const int type = eidx[3];
    for (int p=0; p<2; p++) 
      for (int q=0; q<3; q++) 
        nidxs[p][q] = eidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}

__device__ __host__
inline bool eid2nodes3D_hex(const vf2gpu_hdr_t& h, unsigned int eid, int nidxs[2][3])
{
  const int nodes_idx[3][2][3] = { // 3 types of edges
    {{0, 0, 0}, {1, 0, 0}}, 
    {{0, 0, 0}, {0, 1, 0}}, 
    {{0, 0, 0}, {0, 0, 1}}
  };

  int eidx[4];
  eid2eidx3D_hex(h, eid, eidx);

  if (valid_eidx3D_hex(h, eidx)) {
    const int type = eidx[3];
    for (int p=0; p<2; p++) 
      for (int q=0; q<3; q++) 
        nidxs[p][q] = eidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}

__device__ __host__
inline bool eid2nodes2D(const vf2gpu_hdr_t& h, unsigned int eid, int nidxs[2][2])
{
  const int nodes_idx[2][2][2] = { // 2 types of edges
    {{0, 0}, {1, 0}},
    {{0, 0}, {0, 1}}};

  int eidx[4];
  eid2eidx2D(h, eid, eidx);

  if (valid_eidx2D(h, eidx)) {
    const int type = eidx[2];
    for (int p=0; p<2; p++)
      for (int q=0; q<2; q++)
        nidxs[p][q] = eidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}


__device__ __host__
inline int hexcell_hexfaces(const vf2gpu_hdr_t& h, const int cidx[3], int* fids)
{
  const int fidxs[6][4] = {
    {cidx[0], cidx[1], cidx[2], 0},  
    {cidx[0], cidx[1], cidx[2], 1},
    {cidx[0], cidx[1], cidx[2], 2}, 
    {cidx[0]+1, cidx[1], cidx[2], 0}, 
    {cidx[0], cidx[1]+1, cidx[2], 1},
    {cidx[0], cidx[1], cidx[2]+1, 2}};

  for (int i=0; i<6; i++) 
    fids[i] = fidx2fid3D_hex(h, fidxs[i]);

  return 6;
}

template <typename T>
__device__ __host__
inline void nidx2pos3D(const vf2gpu_hdr_t& h, const int nidx[3], T X[3])
{
  for (int i=0; i<3; i++) 
    X[i] = nidx[i] * h.cell_lengths[i] + h.origins[i];
}

template <typename T>
__device__ __host__
inline void nidx2pos2D(const vf2gpu_hdr_t& h, const int nidx[3], T X[3])
{
  for (int i=0; i<2; i++) 
    X[i] = nidx[i] * h.cell_lengths[i] + h.origins[i];
  X[2] = 0;
}

#endif
