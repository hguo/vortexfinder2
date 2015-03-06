#include "VortexTransitionMatrix.h"
#include <sstream>
#include <cstdio>

VortexTransitionMatrix::VortexTransitionMatrix() :
  _t0(INT_MAX), _t1(INT_MAX),
  _n0(INT_MAX), _n1(INT_MAX)
{
}

VortexTransitionMatrix::VortexTransitionMatrix(int t0, int t1, int n0, int n1) :
  _t0(t0), _t1(t1), 
  _n0(n0), _n1(n1)
{
  _match.resize(_n0*_n1);
}

VortexTransitionMatrix::~VortexTransitionMatrix()
{
}

bool VortexTransitionMatrix::LoadFromFile(const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) return false;

  fread(&_t0, sizeof(int), 1, fp);
  fread(&_t1, sizeof(int), 1, fp);
  fread(&_n0, sizeof(int), 1, fp);
  fread(&_n1, sizeof(int), 1, fp);

  _match.resize(_n0*_n1);
  fread((char*)_match.data(), sizeof(int), _n0*_n1, fp);

  fclose(fp);

  return Valid();
}

bool VortexTransitionMatrix::SaveToFile(const std::string& filename) const
{
  if (!Valid()) return false;

  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) return false;

  fwrite(&_t0, sizeof(int), 1, fp);
  fwrite(&_t1, sizeof(int), 1, fp);
  fwrite(&_n0, sizeof(int), 1, fp);
  fwrite(&_n1, sizeof(int), 1, fp);
  fwrite((char*)_match.data(), sizeof(int), _n0*_n1, fp);

  fclose(fp);
  return true;
}

bool VortexTransitionMatrix::LoadFromFile(const std::string& dataname, int t0, int t1)
{
  return LoadFromFile(MatrixFileName(dataname, t0, t1));
}

bool VortexTransitionMatrix::SaveToFile(const std::string& dataname, int t0, int t1) const 
{
  return SaveToFile(MatrixFileName(dataname, t0, t1));
}

std::string VortexTransitionMatrix::MatrixFileName(const std::string& dataname, int t0, int t1) const
{
  std::stringstream ss;
  ss << dataname << ".match." << t0 << "." << t1;
  return ss.str();
}

int VortexTransitionMatrix::operator()(int i, int j) const
{
  return _match[i*n1() + j];
}

int& VortexTransitionMatrix::operator()(int i, int j)
{
  return _match[i*n1() + j];
}
