#ifndef _FITCURVES_HPP
#define _FITCURVES_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "vector.hpp"

namespace FitCurves {

const int MAX_NPTS = 2048;

// template <int ndims>
// typedef Point<ndims> *BezierCurve<ndims>;

template <int ndims>
inline Vector<ndims> calc_left_tangent(Point<ndims> *pts, int end)
{
	return (pts[end + 1] - pts[end]).normalize();
}

template <int ndims>
inline Vector<ndims> calc_right_tangent(Point<ndims> *pts, int end)
{
	return (pts[end - 1] - pts[end]).normalize();
}

template <int ndims>
inline Vector<ndims> calc_center_tangent(Point<ndims> *pts, int center)
{
	Vector<ndims> v0 = pts[center - 1] - pts[center];
	Vector<ndims> v1 = pts[center] - pts[center + 1];
	return ((v0 + v1) * 0.5).normalize();
}

template <int ndims>
int fit_curves(int npts, Point<ndims> *pts, double error_bound, Point<ndims> *curve, double &sum_error)
{
	Vector<ndims> t_hat1 = calc_left_tangent(pts, 0);
	Vector<ndims> t_hat2 = calc_right_tangent(pts, npts - 1);
	return fit_cubic(pts, 0, npts - 1, t_hat1, t_hat2, error_bound, curve, sum_error);
}

template <int ndims>
void chord_length_parameterize(Point<ndims> *pts, int first, int last, double *u)
{
	u[0] = 0.;
	for (int i = first + 1; i <= last; ++i)
		u[i - first] = u[i - first - 1] + distance(pts[i], pts[i - 1]);
	for (int i = first + 1; i <= last; ++i)
		u[i - first] /= u[last - first];
}

template <int ndims>
Point<ndims> bezier(int deg, Point<ndims> *V, double t)
{
	Point<ndims> *Vtemp = (Point<ndims> *)malloc((deg + 1) * sizeof(Point<ndims>));
	for (int i = 0; i <= deg; ++i) Vtemp[i] = V[i];

	for (int i = 1; i <= deg; ++i)
	{
		for (int j = 0; j <= deg - i; ++j)
		{
			for (int d = 0; d < ndims; ++d)
			{
				Vtemp[j][d] = (1. - t) * Vtemp[j][d] + t * Vtemp[j + 1][d];
			}
		}
	}
	Point<ndims> ret = Vtemp[0];
	free(Vtemp);
	return ret;
}

template <int ndims>
double calc_max_error(Point<ndims> *pts, int first, int last, Point<ndims> *curve, double *u, int &split, double &serr)
{
	split = (last + first + 1) / 2;
	serr = 0;
	double max_dist = 0.;
	for (int i = first + 1; i < last; ++i)
	{
		Point<ndims> P = bezier(3, curve, u[i - first]);
		double dist = distance(P, pts[i]);
		serr += dist * dist;
		if (dist > max_dist)
		{
			max_dist = dist;
			split = i;
		}
	}
	return max_dist * max_dist;
}

template <int ndims>
int fit_cubic(Point<ndims> *pts, int first, int last, Vector<ndims> t_hat1, Vector<ndims> t_hat2, double error_bound, Point<ndims> *curve, double &sum_error)
{
	int npts = last - first + 1;
	if (npts == 2)
	{
		double dist = distance(pts[last], pts[first]) / 3.;
		curve[0] = pts[first];
		curve[3] = pts[last];
		curve[1] = curve[0] + t_hat1 * dist;
		curve[2] = curve[3] + t_hat2 * dist;
		sum_error = 0;
		return 4;
	}

	double *u = (double *)malloc(npts * sizeof(double));
	chord_length_parameterize(pts, first, last, u);
	int nctrlpts = gen_bezier(pts, first, last, u, t_hat1, t_hat2, curve);
	int split;
	double max_error = calc_max_error(pts, first, last, curve, u, split, sum_error);
	if (max_error < error_bound)
	{
		free(u);
		return nctrlpts;
	}

	int max_n_iters = 4;
	if (max_error < error_bound * error_bound)
	{
		for (int i = 0; i < max_n_iters; ++i)
		{
			double *u_prime = (double *)malloc(npts * sizeof(double));
			reparameterize(pts, first, last, u, curve, u_prime);
			nctrlpts = gen_bezier(pts, first, last, u_prime, t_hat1, t_hat2, curve);
			max_error = calc_max_error(pts, first, last, curve, u_prime, split, sum_error);
			free(u);
			u = u_prime;
		}
		if (max_error < error_bound)
		{
			free(u);
			return nctrlpts;
		}
	}
	free(u);

	Vector<ndims> t_hat_center = calc_center_tangent(pts, split);
	double serror1, serror2;
	int nctrlpts0 = fit_cubic(pts, first, split, t_hat1, t_hat_center, error_bound, curve, serror1);
	int nctrlpts1 = fit_cubic(pts, split, last, -t_hat_center, t_hat2, error_bound, curve + nctrlpts0, serror2);
	sum_error = serror1 + serror2;
	return nctrlpts0 + nctrlpts1;
}

/*
 *  B0, B1, B2, B3 :
 *	Bezier multipliers
 */
static double B0(double u)
{
    double tmp = 1.0 - u;
    return (tmp * tmp * tmp);
}


static double B1(double u)
{
    double tmp = 1.0 - u;
    return (3 * u * (tmp * tmp));
}

static double B2(double u)
{
    double tmp = 1.0 - u;
    return (3 * u * u * tmp);
}

static double B3(double u)
{
    return (u * u * u);
}


template <int ndims>
int gen_bezier(Point<ndims> *pts, int first, int last, double *u_prime, Vector<ndims> t_hat1, Vector<ndims> t_hat2, Point<ndims> *curve)
{
	int npts = last - first + 1;

	Vector<ndims> A[MAX_NPTS][2];
	for (int i = 0; i < npts; ++i)
	{
		A[i][0] = t_hat1 * B1(u_prime[i]);
		A[i][1] = t_hat2 * B2(u_prime[i]);
	}

	double C[2][2] = {0., 0., 0., 0., };
	double X[2] = {0., 0., };

	for (int i = 0; i < npts; ++i)
	{
		C[0][0] += dot(A[i][0], A[i][0]);
		C[0][1] += dot(A[i][0], A[i][1]);
		C[1][0] = C[0][1];
		C[1][1] += dot(A[i][1], A[i][1]);

		Vector<ndims> tmp = pts[first + i] -
			(pts[first] * B0(u_prime[i]) + pts[first] * B1(u_prime[i]) + pts[last] * B2(u_prime[i]) + pts[last] * B3(u_prime[i]));
		X[0] += dot(A[i][0], tmp);
		X[1] += dot(A[i][1], tmp);
	}

	/* Compute the determinants of C and X	*/
	double det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1];
	double det_C0_X  = C[0][0] * X[1]    - C[1][0] * X[0];
	double det_X_C1  = X[0]    * C[1][1] - X[1]    * C[0][1];

	/* Finally, derive alpha values	*/
	double alpha_l = (det_C0_C1 == 0) ? 0.0 : det_X_C1 / det_C0_C1;
	double alpha_r = (det_C0_C1 == 0) ? 0.0 : det_C0_X / det_C0_C1;

	/* If alpha negative, use the Wu/Barsky heuristic (see text) */
	/* (if alpha is 0, you get coincident control points that lead to
	 * divide by zero in any subsequent NewtonRaphsonRootFind() call. */
	double seg_length = distance(pts[last], pts[first]);
	double epsilon = 1.e-6 * seg_length;

	if (alpha_l < epsilon || alpha_r < epsilon)
	{
		/* fall back on standard (probably inaccurate) formula, and subdivide further if needed. */
		double dist = seg_length / 3.0;
		curve[0] = pts[first];
		curve[3] = pts[last];
		curve[1] = t_hat1 * dist + curve[0];
		curve[2] = t_hat2 * dist + curve[3];
		return 4;
	}

	/*  First and last control points of the Bezier curve are */
	/*  positioned exactly at the first and last data points */
	/*  Control points 1 and 2 are positioned an alpha distance out */
	/*  on the tangent vectors, left and right, respectively */
	curve[0] = pts[first];
	curve[3] = pts[last];
	curve[1] = t_hat1 * alpha_l + curve[0];
	curve[2] = t_hat2 * alpha_r + curve[3];
	return 4;
}

template <int ndims>
void reparameterize(Point<ndims> *pts, int first, int last, double *u, Point<ndims> *curve, double *u_prime)
{
	int npts = last - first + 1;
	for (int i = first; i <= last; ++i)
		u_prime[i - first] = NewtonRaphsonRootFind(curve, pts[i], u[i-first]);
}

template <int ndims>
double NewtonRaphsonRootFind(Point<ndims> *Q, Point<ndims> P, double u)
{
	/* Compute Q(u)	*/
	Point<ndims> Q_u = bezier(3, Q, u);

	/* Generate control vertices for Q'	*/
	Point<ndims> Q1[3];
	for (int i = 0; i <= 2; i++) Q1[i] = (Q[i + 1] - Q[i]) * 3.;

	/* Generate control vertices for Q'' */
	Point<ndims> Q2[2];
	for (int i = 0; i <= 1; i++) Q2[i] = (Q1[i + 1] - Q1[i]) * 2.;

	/* Compute Q'(u) and Q''(u)	*/
	Point<ndims> Q1_u = bezier(2, Q1, u);
	Point<ndims> Q2_u = bezier(1, Q2, u);

	/* Compute f(u)/f'(u) */
	double numerator = 0;
	double denominator = 0;
	for (int i = 0; i < ndims; ++i)
	{
		numerator += (Q_u[i] - P[i]) * Q1_u[i];
		denominator += Q1_u[i] * Q1_u[i] +  (Q_u[i] - P[i]) * Q2_u[i];
	}
	if (denominator == 0.0f) return u;

	/* u = u - f(u)/f'(u) */
	double u_prime = u - (numerator/denominator);
	return u_prime;
}

}

#endif
