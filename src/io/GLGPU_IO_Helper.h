#ifndef _GLGPU_IO_HELPER_H
#define _GLGPU_IO_HELPER_H

#include "GLHeader.h"
#include "BDATReader.h"

bool GLGPU_IO_Helper_ReadBDAT(
    const std::string& filename, 
    GLHeader &hdr, 
    double **re, 
    double **im, 
    bool header_only=false);

bool GLGPU_IO_Helper_ReadLegacy(
    const std::string& filename, 
    GLHeader &hdr, 
    double **re, 
    double **im, 
    bool header_only=false);

bool GLGPU_IO_Helper_ReadNetCDF(
    const std::string& filename, 
    GLHeader &hdr, 
    double **re, 
    double **im);

bool GLGPU_IO_Helper_WriteNetCDF(
    const std::string& filename, 
    GLHeader &hdr, 
    double *re, 
    double *im);

#endif
