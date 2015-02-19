#ifndef _GLGPU_IO_HELPER_H
#define _GLGPU_IO_HELPER_H

#include "BDATReader.h"

bool GLGPU_IO_Helper_ReadBDAT(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double *B,
    double &Jxext, 
    double &Kx, 
    double &V, 
    double **re, 
    double **im);

bool GLGPU_IO_Helper_ReadLegacy(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double *B,
    double &Jxext, 
    double &Kx, 
    double &V, 
    double **re, 
    double **im);

bool GLGPU_IO_Helper_ReadNetCDF(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double *B,
    double &Jxext, 
    double &Kx, 
    double &V, 
    double **re, 
    double **im);

bool GLGPU_IO_Helper_WriteNetCDF(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double *B,
    double &Jxext, 
    double &Kx, 
    double &V, 
    double **re, 
    double **im);

#endif
