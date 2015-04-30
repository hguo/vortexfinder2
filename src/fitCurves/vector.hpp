#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_

#include <cmath>

namespace FitCurves {

template <int ndims>
struct Vector
{
	double coords[ndims];

	Vector() {}
	Vector(const Vector &p)
	{
		for (int i = 0; i < ndims; ++i) coords[i] = p[i];
	}
	Vector& operator=(const Vector &p)
	{
		for (int i = 0; i < ndims; ++i) coords[i] = p[i];
		return *this;
	}

	double operator[](int index) const
	{
		return coords[index];
	}

	double& operator[](int index)
	{
		return coords[index];
	}

	Vector<ndims> operator+(const Vector<ndims> &p) const
	{
		Vector<ndims> ret(*this);
		for (int i = 0; i < ndims; ++i) ret[i] += p[i];
		return ret;
	}

	Vector<ndims> operator-(const Vector<ndims> &p) const
	{
		Vector<ndims> ret(*this);
		for (int i = 0; i < ndims; ++i) ret[i] -= p[i];
		return ret;
	}

	Vector<ndims> operator*(double f) const
	{
		Vector<ndims> ret(*this);
		for (int i = 0; i < ndims; ++i) ret[i] *= f;
		return ret;
	}

	Vector<ndims> operator-() const
	{
		Vector<ndims> ret(*this);
		for (int i = 0; i < ndims; ++i) ret[i] = -ret[i];
		return ret;
	}

	double length() const
	{
		double ret = 0;
		for (int i = 0; i < ndims; ++i) ret += coords[i] * coords[i];
		return sqrt(ret);
	}

	Vector normalize() const
	{
		Vector ret(*this);
		double len = length();
		for (int i = 0; i < ndims; ++i) ret.coords[i] /= len;
		return ret;
	}
};

#define Point Vector	// Treat Point as Vector

template <int ndims>
inline double dot(const Vector<ndims> &v0, const Vector<ndims> &v1)
{
	double ret = 0;
	for (int i = 0; i < ndims; ++i) ret += v0[i] * v1[i];
	return ret;
}

template <int ndims>
inline double distance(const Vector<ndims> &v0, const Vector<ndims> &v1)
{
	return (v0 - v1).length();
}

}

#endif	// _VECTOR_HPP_
