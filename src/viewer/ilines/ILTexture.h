/* $Id: ILTexture.h,v 1.20 2005/10/19 10:52:40 ovidiom Exp $ */

/* forward declarations */
namespace ILines { class ILTexture; }

#ifndef _ILTEXTURE_H_
#define _ILTEXTURE_H_

#include <cstdlib>
#include <cmath>

#include "ILLightingModel.h"

#include "Vector.h"


namespace ILines
{
	#define PI_DOUBLE	3.141592653589793


	/**
	 * @brief Provides functions to compute lighting textures for illuminating lines.
	 *
	 * This class computes two-dimensional lighting textures which can then be
	 * used for the illumination of lines using texture mapping. These lighting
	 * textures are used by the ILRender class for rendering.
	 */
	class ILTexture
	{
	public:
		/** @brief Computes lighting textures for illuminating lines. */
		static void computeTextures(float ka, float kd, float ks, float n,
		                            int texDim,
		                            float *tex, float *texDiff, float *texSpec,
		                            ILLightingModel::Model lightingModel,
		                            bool stretch = false,
		                            const float *L = NULL, const float *V = NULL);
                            
	private:
		/** @name Functions for the maximum principle Phong lighting model. */
		/*@{*/
		/** @brief Computes the diffuse term of the lighting model. */
		static double computeDiffTermMaximum(double LT);

		/** @brief Computes the specular term of the lighting model. */
		static double computeSpecTermMaximum(double LT, double VT, double n);
		/*@}*/

		/** @name Functions for the cylinder averaging lighting models. */
		/*@{*/
		/** @brief Computes the angle alpha. */
		static double computeAlpha(double LV, double LT, double VT);

		/** @brief Computes the diffuse term of the lighting model. */
		static double computeDiffTerm(double LT, double alpha);

		/** @brief Computes the specular term of the lighting model. */
		static double computeSpecTermPhong(double LT, double VT,
		                                   double alpha, double n);

		/** @brief Computes the integrand needed for the specular lighting. */
		static double computeSpecTermPhongIntegrand(double LT, double VT,
		                                            double alpha, double n,
		                                            double theta);

		/** @brief Computes a part of the specular term of the lighting model. */
		static double computeSpecTermBlinn(double alpha, double beta, double n);

		/** @brief Computes the integrand needed for the specular lighting. */
		static double computeSpecTermBlinnIntegrand(double beta, double n,
		                                            double theta);

		/** @brief Computes the stretching factor for the diffuse lighting. */
		static double computeStretchFactorDiff();

		/** @brief Computes the stretching factor for the specular lighting. */
		static double computeStretchFactorSpecBlinn(int n);
		/*@}*/
	};
}

#endif /* _ILTEXTURE_H_ */

