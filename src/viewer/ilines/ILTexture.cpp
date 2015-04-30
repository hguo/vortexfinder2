/* $Id: ILTexture.cpp,v 1.28 2005/10/19 10:52:40 ovidiom Exp $ */

#include "ILTexture.h"


namespace ILines
{
	/**
	 * This function computes two-dimensional lighting textures which can
	 * be used for illuminating lines.
	 *
	 * @param ka             The ambient reflection coefficient.
	 * @param kd             The diffuse reflection coefficient.
	 * @param ks             The specular reflection coefficient.
	 * @param n              The gloss exponent for the specular lighting.
	 * @param texDim         The dimension of the \e square matrices.
	 * @param tex            Address where to store the texture for the
	 *                       full lighting. \n
	 *                       If NULL, this texture will not be computed.
	 * @param texDiff        Address where to store the texture for the ambient and
	 *                       diffuse lighting. \n
	 *                       If NULL, this texture will not be computed.
	 * @param texSpec        Address where to store the texture for the specular
	 *                       lighting. \n
	 *                       If NULL, this texture will not be computed.
	 * @param lightingModel  The lighting model to use.
	 * @param stretch        Flag whether to stretch the dynamic range.
	 * @param L              The \e normalized light vector. \n
	 *                       Only required for the ILLightingModel::IL_CYLINDER_PHONG
	 *                       lighting model.
	 * @param V              The \e normalized viewing vector.
	 *                       Only required for the ILLightingModel::IL_CYLINDER_PHONG
	 *                       lighting model.
	 */
	void ILTexture::computeTextures(float ka, float kd, float ks, float n,
	                                int texDim,
	                                float *tex, float *texDiff, float *texSpec,
	                                ILLightingModel::Model lightingModel,
	                                bool stretch,
	                                const float *L, const float *V)
	{
		float	*iter, *iterDiff, *iterSpec;
		double	s, t;
		double	LT, VT, LV;
		double	alpha, beta;
		double	diffTerm, specTerm;
		double	stretchDiff, stretchSpec;

		iter = tex;
		iterDiff = texDiff;
		iterSpec = texSpec;

		stretchDiff = stretchSpec = 1.0;
		if (stretch)
		{
			if (lightingModel == ILLightingModel::IL_CYLINDER_BLINN)
			{
				stretchDiff = computeStretchFactorDiff();
				stretchSpec = computeStretchFactorSpecBlinn((int)n);
			}
			else if (lightingModel == ILLightingModel::IL_CYLINDER_PHONG)
				stretchDiff = computeStretchFactorDiff();
		}

		for (int y = 0; y < texDim; y++)
			for (int x = 0; x < texDim; x++)
			{
				/* Avoid divisions by zero. */
				s = ((double)x + 0.5) / (double)texDim;
				t = ((double)y + 0.5) / (double)texDim;

				switch (lightingModel)
				{
				case ILLightingModel::IL_MAXIMUM_PHONG:
					LT = 2.0 * s - 1.0;
					VT = 2.0 * t - 1.0;

					diffTerm = computeDiffTermMaximum(LT);
					specTerm = computeSpecTermMaximum(LT, VT, n);
					break;
				case ILLightingModel::IL_CYLINDER_BLINN:
					alpha = acos(2.0 * s - 1.0);
					LT = 2.0 * t - 1.0;
					beta = acos(2.0 * t - 1.0);

					diffTerm = stretchDiff * computeDiffTerm(LT, alpha);
					specTerm = stretchSpec * computeSpecTermBlinn(alpha, beta, n);
					break;
				case ILLightingModel::IL_CYLINDER_PHONG:
					LT = 2.0 * s - 1.0;
					VT = 2.0 * t - 1.0;
					LV = dot(L, V);

					alpha = computeAlpha(LV, LT, VT);

					diffTerm = stretchDiff * computeDiffTerm(LT, alpha);
					specTerm = computeSpecTermPhong(LT, VT, alpha, n);
					break;
				default:
					return;
				}

				if (iter != NULL)
					*iter++ = (float)(ka + kd * diffTerm + ks * specTerm);
				if (iterDiff != NULL)
					*iterDiff++ = (float)(ka + kd * diffTerm);
				if (iterSpec != NULL)
					*iterSpec++ = (float)(ks * specTerm);
			}
	}

	/**
	 * The diffuse term of the ILLightingModel::IL_MAXIMUM_PHONG lighting model
	 * is computed with the given parameters as
	 * \f[
	 * \mathbf{L}\cdot\mathbf{N}_{\alpha} = \sqrt{1-(\mathbf{L}\cdot\mathbf{T})^2}.
	 * \f]
	 *
	 * @param LT  The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{T} \f$.
	 * @return    The diffuse term of the ILLightingModel::IL_MAXIMUM_PHONG
	 *            lighting model.
	 */
	double ILTexture::computeDiffTermMaximum(double LT)
	{
		return (sqrt(1.0 - LT * LT));
	}

	/**
	 * The specular term of the ILLightingModel::IL_MAXIMUM_PHONG lighting model
	 * is computed with the given parameters as
	 * \f[
	 * (\mathbf{V}\cdot\mathbf{R}_{\alpha/2})^{n}
	 *                                 = \left(-(\mathbf{V}\cdot\mathbf{T})
	 *                                          (\mathbf{L}\cdot\mathbf{T})
	 *                                         +\sqrt{1-(\mathbf{V}\cdot\mathbf{T})^2}
	 *                                          \sqrt{1-(\mathbf{L}\cdot\mathbf{T})^2}
	 *                                   \right)^{n}.
	 * \f]
	 *
	 * @param LT  The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{T} \f$.
	 * @param VT  The dot product of \f$ \mathbf{V} \f$ and \f$ \mathbf{T} \f$.
	 * @param n   The gloss exponent for the specular lighting.
	 * @return    The specular term of the ILLightingModel::IL_MAXIMUM_PHONG
	 *            lighting model.
	 */
	double ILTexture::computeSpecTermMaximum(double LT, double VT, double n)
	{
		double	VR;

		VR = -VT * LT + sqrt((1.0 - VT * VT) * (1.0 - LT * LT));

		if (VR < 0.0)
			VR = 0.0;

		return (pow(VR, n));
	}

	/**
	 * Computes the angle alpha with the given parameters as
	 * \f[
	 * \alpha = \arccos\left(
	 *                       \frac{\mathbf{L}\cdot\mathbf{V}
	 *                             -(\mathbf{V}\cdot\mathbf{T})
	 *                              (\mathbf{L}\cdot\mathbf{T})}
	 *                            {\sqrt{1-(\mathbf{V}\cdot\mathbf{T})^2}
	 *                             \sqrt{1-(\mathbf{L}\cdot\mathbf{T})^2}}
	 *                 \right).
	 * \f]
	 *
	 * @param LV  The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{V} \f$.
	 * @param LT  The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{T} \f$.
	 * @param VT  The dot product of \f$ \mathbf{V} \f$ and \f$ \mathbf{T} \f$.
	 * @return    The angle \f$ \alpha \f$.
	 */
	double ILTexture::computeAlpha(double LV, double LT, double VT)
	{
		double	cosAlpha;

		cosAlpha = (LV - VT * LT) / sqrt((1.0 - VT * VT) * (1.0 - LT * LT));

		/* Sometimes necessary due to numerical errors. */
		if (cosAlpha > +1.0)
			cosAlpha = +1.0;
		if (cosAlpha < -1.0)
			cosAlpha = -1.0;

		return (acos(cosAlpha));
	}

	/**
	 * The diffuse term of the ILLightingModel::IL_CYLINDER_PHONG
	 * and ILLightingModel::IL_CYLINDER_BLINN lighting models is
	 * computed with the given parameters as
	 * \f[
	 * \int\limits_{\alpha-\pi/2}^{\pi/2}
	 *     (\mathbf{L}\cdot\mathbf{N}_{\theta})\frac{\cos\theta}{2}d\theta
	 *     = \sqrt{1-(\mathbf{L}\cdot\mathbf{T})^2}
	 *       \frac{\sin\alpha+(\pi-\alpha)\cos\alpha}{4}.
	 * \f]
	 *
	 * @param LT     The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{T} \f$.
	 * @param alpha  The angle between the projections of \f$ \mathbf{L} \f$
	 *               and \f$ \mathbf{V} \f$ onto the normal space of the line.
	 * @return       The diffuse term of the ILLightingModel::IL_CYLINDER_PHONG and
	 *               ILLightingModel::IL_CYLINDER_BLINN lighting models.
	 */
	double ILTexture::computeDiffTerm(double LT, double alpha)
	{
		double	LN;

		LN =   sqrt(1.0 - LT * LT)
		     * (sin(alpha) + (PI_DOUBLE - alpha) * cos(alpha)) / 4.0;

		return (LN);
	}

	/**
	 * The specular term of the ILLightingModel::IL_CYLINDER_PHONG
	 * lighting model is computed with the given parameters as
	 * \f[
	 * \int\limits_{\alpha-\pi/2}^{\pi/2}
	 *     (\mathbf{V}\cdot\mathbf{R}_{\theta})^{n}\frac{\cos\theta}{2}d\theta,
	 * \f]
	 * where
	 * \f[
	 * \mathbf{V}\cdot\mathbf{R}_{\theta} = -(\mathbf{V}\cdot\mathbf{T})
	 *                                       (\mathbf{L}\cdot\mathbf{T})
	 *                                      +\sqrt{1-(\mathbf{V}\cdot\mathbf{T})^2}
	 *                                       \sqrt{1-(\mathbf{L}\cdot\mathbf{T})^2}
	 *                                       \cos(2\theta-\alpha).
	 * \f]
	 *
	 * @param LT     The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{T} \f$.
	 * @param VT     The dot product of \f$ \mathbf{V} \f$ and \f$ \mathbf{T} \f$.
	 * @param alpha  The angle \f$ \alpha \f$.
	 * @param n      The gloss exponent of the specular lighting.
	 * @return       The specular term of the ILLightingModel::IL_CYLINDER_PHONG
	 *               lighting model.
	 */
	double ILTexture::computeSpecTermPhong(double LT, double VT,
	                                       double alpha, double n)
	{
		double	a, b;
		int		m;
		double	h;
		double	xi;
		double	integral;

		a = alpha - PI_DOUBLE / 2.0;
		b = PI_DOUBLE / 2.0;

		m = 10;
		h = (b - a) / (2.0 * m);

		integral = 0.0;
		for (int i = 0; i < 2 * m; i += 2)
		{
			xi = a + i * h;
			integral += 2.0 * computeSpecTermPhongIntegrand(LT, VT, alpha, n, xi);
			xi = a + (i + 1) * h;
			integral += 4.0 * computeSpecTermPhongIntegrand(LT, VT, alpha, n, xi);
		}
		integral += computeSpecTermPhongIntegrand(LT, VT, alpha, n, b);

		/* f(a) has been accounted for twice inside the for-loop. */
		integral -= computeSpecTermPhongIntegrand(LT, VT, alpha, n, a);

		integral *= h / 3.0;

		return (integral);
	}

	/**
	 * This function computes the integrand needed for the specular term
	 * of the ILLightingModel::IL_CYLINDER_PHONG lighting model. \n
	 * The integrand is
	 * \f[
	 * (\mathbf{V}\cdot\mathbf{R}_{\theta})^{n}\frac{\cos\theta}{2},
	 * \f]
	 * where
	 * \f[
	 * \mathbf{V}\cdot\mathbf{R}_{\theta} = -(\mathbf{V}\cdot\mathbf{T})
	 *                                       (\mathbf{L}\cdot\mathbf{T})
	 *                                      +\sqrt{1-(\mathbf{V}\cdot\mathbf{T})^2}
	 *                                       \sqrt{1-(\mathbf{L}\cdot\mathbf{T})^2}
	 *                                       \cos(2\theta-\alpha).
	 * \f]
	 *
	 * @param LT     The dot product of \f$ \mathbf{L} \f$ and \f$ \mathbf{T} \f$.
	 * @param VT     The dot product of \f$ \mathbf{V} \f$ and \f$ \mathbf{T} \f$.
	 * @param alpha  The angle \f$ \alpha \f$.
	 * @param n      The gloss exponent of the specular lighting.
	 * @param theta  The angle \f$ \theta \f$.
	 * @return       The value of the integrand for for the given parameters.
	 */
	double ILTexture::computeSpecTermPhongIntegrand(double LT, double VT,
	                                                double alpha, double n,
	                                                double theta)
	{
		double	VR;

		VR =   -VT * LT + sqrt((1.0 - VT * VT) * (1.0 - LT * LT))
		     * cos(2.0 * theta - alpha);

		if (VR < 0.0)
			VR = 0.0;

		return (pow(VR, n) * (cos(theta) / 2.0));
	}

	/**
	 * A part of the specular term of the ILLightingModel::IL_CYLINDER_BLINN
	 * lighting model is computed with the given parameters as
	 * \f[
	 * \int\limits_{\alpha-\pi/2}^{\pi/2}
	 *     (\mathbf{H}\cdot\mathbf{N}_{\theta})^{n}\frac{\cos\theta}{2}d\theta
	 *     = \int\limits_{\alpha-\pi/2}^{\pi/2}
	 *       \cos^n(\theta-\beta)\frac{\cos\theta}{2}d\theta.
	 * \f]
	 *
	 * @param alpha  The angle \f$ \alpha \f$.
	 * @param beta   The angle \f$ \beta \f$.
	 * @param n      The gloss exponent of the specular lighting.
	 * @return       The specular term of the ILLightingModel::IL_CYLINDER_BLINN
	 *               lighting model.
	 */
	double ILTexture::computeSpecTermBlinn(double alpha, double beta, double n)
	{
		double	a, b;
		int		m;
		double	h;
		double	xi;
		double	integral;

		a = alpha - PI_DOUBLE / 2.0;
		b = PI_DOUBLE / 2.0;

		m = 10;
		h = (b - a) / (2.0 * m);

		integral = 0.0;
		for (int i = 0; i < 2 * m; i += 2)
		{
			xi = a + i * h;
			integral += 2.0 * computeSpecTermBlinnIntegrand(beta, n, xi);
			xi = a + (i + 1) * h;
			integral += 4.0 * computeSpecTermBlinnIntegrand(beta, n, xi);
		}
		integral += computeSpecTermBlinnIntegrand(beta, n, b);

		/* f(a) has been accounted for twice inside the for-loop. */
		integral -= computeSpecTermBlinnIntegrand(beta, n, a);

		integral *= h / 3.0;

		return (integral);
	}

	/**
	 * This function computes the integrand needed for the specular term
	 * of the ILLightingModel::IL_CYLINDER_BLINN lighting model. \n
	 * The integrand is
	 * \f[
	 * \cos^{n}(\theta-\beta)\frac{\cos\theta}{2}.
	 * \f]
	 *
	 * @param beta   The angle \f$ \beta \f$.
	 * @param n      The gloss exponent of the specular lighting.
	 * @param theta  The angle \f$ \theta \f$.
	 * @return       The value of the integrand for for the given parameters.
	 */
	double ILTexture::computeSpecTermBlinnIntegrand(double beta, double n,
	                                                double theta)
	{
		double	y;

		y = cos(theta - beta);

		if (y < 0.0)
			y = 0.0;

		return (pow(y, n) * (cos(theta) / 2.0));
	}

	/**
	 * The stretching factor for the diffuse term of the
	 * ILLightingModel::IL_CYLINDER_PHONG and ILLightingModel::IL_CYLINDER_BLINN
	 * lighting models is
	 * \f[
	 * \frac{4}{\pi}.
	 * \f]
	 *
	 * @return  The stretching factor for the diffuse term of the
	 *          ILLightingModel::IL_CYLINDER_PHONG and
	 *          ILLightingModel::IL_CYLINDER_BLINN lighting models.
	 */
	double ILTexture::computeStretchFactorDiff()
	{
		return (4.0 / PI_DOUBLE);
	}

	/**
	 * The stretching factor for the specular term of the
	 * ILLightingModel::IL_CYLINDER_BLINN lighting model is
	 * \f[
	 * \frac{2}{\sqrt{\pi}}
	 * \Gamma\left(\frac{n+3}{2}\right)
	 * \left/\Gamma\left(\frac{n+1}{2}\right)\right.,
	 * \f]
	 * which simplifies to
	 * \f[
	 * \frac{3}{2}\frac{5}{4}\frac{7}{6}\cdots\frac{n+1}{n}
	 * \f]
	 * for (positive) even \f$ n \f$ and to
	 * \f[
	 * \frac{2}{\pi}\frac{2}{1}\frac{4}{3}\frac{6}{5}\cdots\frac{n+1}{n}
	 * \f]
	 * for odd \f$ n \f$.
	 *
	 * @param n  The gloss exponent of the specular lighting.
	 * @return   The stretching factor for the specular term of the
	 *           ILLightingModel::IL_CYLINDER_BLINN lighting model.
	 */
	double ILTexture::computeStretchFactorSpecBlinn(int n)
	{
		double	factor;

		factor = 1.0;
		while (n > 0)
		{
			factor *= (double)(n + 1) / (double)n;
			n -= 2;
		}

		return ((n % 2) ? factor : ((2.0 / PI_DOUBLE) * factor));
	}
}

