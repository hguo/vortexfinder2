/* $Id: ILUtilities.h,v 1.17 2005/10/17 10:12:09 ovidiom Exp $ */

/* forward declarations */
namespace ILines { class ILUtilities; }

#ifndef _ILUTILITIES_H_
#define _ILUTILITIES_H_

#include <cmath>
#include <climits>
#include <algorithm>

#include <GL/glew.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "Vector.h"


namespace ILines
{
	#define PI_FLOAT	3.141592653589f


	/**
	 * @brief Provides some general functions to be used by other classes.
	 */
	class ILUtilities
	{
	public:
		/** @brief Computes the rotational component of an OpenGL scene. */
		static float *computeRotationMatrix();

		/** @brief Performs z-sorting on a set of given line strips. */
		static int *zsort(int *first, int *vertCount, int lineCount,
		                  const float *homVertices);

	private:
		/** @brief Helper class for z-sorting with the std::sort STL function. */
		struct DepthSorter;

		/** @brief Core sorting algorithm for the z-sorting. */
		static void qsort(int *left, int *right, int *depths);

		/** @brief Computes the depth for some point in an OpenGL scene. */
		static int getDepth(const GLfloat *v, GLfloat *mvp);
	};
}

#endif /* _ILUTILITIES_H_ */

