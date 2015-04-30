/* $Id: Vector.h,v 1.5 2005/10/17 10:12:09 ovidiom Exp $ */

/* forward declarations */
namespace ILines { struct Vector3f; }

#ifndef _VECTOR3F_H_
#define _VECTOR3F_H_

#include <cmath>


namespace ILines
{
	struct Vector3f
	{
		float	x, y, z;

		Vector3f()
		{
			x = y = z = 0.0f;
		}

		Vector3f(float x, float y, float z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}

		Vector3f(float xyz)
		{
			x = y = z = xyz;
		}

		Vector3f(const float *xyzArr)
		{
			x = xyzArr[0];
			y = xyzArr[1];
			z = xyzArr[2];
		}

		operator const float *() const
		{
			return ((const float *)&x);
		}

		float &operator[](unsigned int idx)
		{
			return (*(((float *)&x) + idx));
		}

		void operator +=(float s)
		{
			x += s;
			y += s;
			z += s;
		}

		void operator +=(const Vector3f &v)
		{
			x += v.x;
			y += v.y;
			z += v.z;
		}

		void operator -=(float s)
		{
			x -= s;
			y -= s;
			z -= s;
		}

		void operator -=(const Vector3f &v)
		{
			x -= v.x;
			y -= v.y;
			z -= v.z;
		}

		void operator *=(float s)
		{
			x *= s;
			y *= s;
			z *= s;
		}

		void operator *=(const Vector3f &v)
		{
			x *= v.x;
			y *= v.y;
			z *= v.z;
		}

		void operator /=(float s)
		{
			float	inv = 1.0f / s;

			x *= inv;
			y *= inv;
			z *= inv;
		}

		void operator /=(const Vector3f &v)
		{
			x /= v.x;
			y /= v.y;
			z /= v.z;
		}
	};

	Vector3f operator +(const Vector3f &v, float s);
	Vector3f operator +(float s, const Vector3f &v);
	Vector3f operator +(const Vector3f &u, const Vector3f &v);
	Vector3f operator -(const Vector3f &v, float s);
	Vector3f operator -(float s, const Vector3f &v);
	Vector3f operator -(const Vector3f &u, const Vector3f &v);
	Vector3f operator *(const Vector3f &v, float s);
	Vector3f operator *(float s, const Vector3f &v);
	Vector3f operator *(const Vector3f &u, const Vector3f &v);
	Vector3f operator /(const Vector3f &v, float s);
	Vector3f operator /(float s, const Vector3f &v);
	Vector3f operator /(const Vector3f &u, const Vector3f &v);
	Vector3f operator -(const Vector3f &v);
	float dot(const Vector3f &u, const Vector3f &v);
	Vector3f cross(const Vector3f &u, const Vector3f &v);
	float length(const Vector3f &v);
	Vector3f normalize(const Vector3f &v);
}

#endif /* _VECTOR3F_H_ */

