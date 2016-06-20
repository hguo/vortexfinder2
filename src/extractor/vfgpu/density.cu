
// recycled code

////////////////////////////////////////// density estimation
__device__
inline static float gaussian(float x2, float sigma2) // x^2, sigma^2
{
  return exp(-0.5 * x2 / sigma2);
}

__device__
inline static float step(float x2, float h2)
{
  return x2 <= h2;
}

__global__
static void density_estimate(
    const gpu_hdr_t *h,
    int npts, 
    int nlines, 
    const float *pts, 
    const int *acc, float *volume)
{
  const int nidx[3] = {
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z};
  if (!valid_nidx(*h, nidx)) return;

  const int nid = nidx2nid(*h, nidx);
  float X[3];
  nidx2pos(*h, nidx, X);

  float density = 0;
  for (int i=0; i<npts; i++) {
    float d2 = dist2(X, pts + i*3);
    density += gaussian(d2, 1);
    // density += step(d2, 1);
  }
  density = density / npts;

  volume[nid] = density; 
}


void vfgpu_density_estimate(int npts, int nlines, const float *pts, const int *acc)
{
  fprintf(stderr, "npts=%d, nlines=%d\n", npts, nlines);

  float *d_pts;
  int *d_acc;

  cudaMalloc((void**)&d_pts, sizeof(float)*npts*3);
  cudaMalloc((void**)&d_acc, sizeof(int)*nlines);

  if (density == NULL)
    density = (int*)malloc(sizeof(int)*h[0].count);
  if (d_density == NULL) 
    cudaMalloc((void**)&d_density, sizeof(int)*h[0].count);

  cudaMemcpy(d_pts, pts, sizeof(float)*npts*3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc, acc, sizeof(int)*nlines, cudaMemcpyHostToDevice);

  const dim3 volumeSize = dim3(h[0].d[0], h[0].d[1], h[0].d[2]);
  const dim3 blockSize = dim3(16, 8, 2);
  const dim3 gridSize = idivup(volumeSize, blockSize);

  density_estimate<<<gridSize, blockSize>>>(d_h[0], npts, nlines, d_pts, d_acc, d_density);

  cudaMemcpy(density, d_density, sizeof(float)*h[0].count, cudaMemcpyDeviceToHost);

#ifdef WITH_NETCDF
  int ncid;
  int dimids[3];
  int varids[1];

  size_t starts[3] = {0, 0, 0}, 
         sizes[3] = {h[0].d[2], h[0].d[1], h[0].d[0]};

  NC_SAFE_CALL( nc_create("density.nc", NC_CLOBBER | NC_64BIT_OFFSET, &ncid) );
  NC_SAFE_CALL( nc_def_dim(ncid, "z", sizes[0], &dimids[0]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "y", sizes[1], &dimids[1]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "x", sizes[2], &dimids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "density", NC_FLOAT, 3, dimids, &varids[0]) );
  NC_SAFE_CALL( nc_enddef(ncid) );

  NC_SAFE_CALL( nc_put_vara_float(ncid, varids[0], starts, sizes, density) );
  NC_SAFE_CALL( nc_close(ncid) );
#endif

  cudaFree(d_pts);
  cudaFree(d_acc);
  
  checkLastCudaError("density estimate");
}
