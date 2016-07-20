#ifndef _GLGPU_IO_HELPER_H
#define _GLGPU_IO_HELPER_H

#include "GLHeader.h"
#include "BDATReader.h"

bool GLGPU_IO_Helper_ReadBDAT(
    const std::string& filename, 
    GLHeader &hdr,
    float **rho, float **phi, float **re, float **im, float **J,
    bool header_only=false);

bool GLGPU_IO_Helper_ReadLegacy(
    const std::string& filename, 
    GLHeader &hdr, 
    float **rho, float **phi, float **re, float **im, float **J,
    bool header_only=false);

void GLGPU_IO_Helper_ComputeSupercurrent(
    GLHeader &h, const float *re_im, float **J);

bool GLGPU_IO_Helper_ReadNetCDF(
    const std::string& filename, 
    GLHeader &hdr, 
    float **psi); 

bool GLGPU_IO_Helper_WriteNetCDF(
    const std::string& filename, 
    GLHeader &hdr, 
    float **psi); 

#endif
