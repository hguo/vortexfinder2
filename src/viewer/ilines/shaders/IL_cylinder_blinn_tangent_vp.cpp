/* $Id: IL_cylinder_blinn_tangent_vp.cpp,v 1.5 2005/10/17 10:41:21 ovidiom Exp $ */

/**
 * @file  IL_cylinder_blinn_tangent_vp.cpp
 * @brief Contains a vertex program in string representation.
 */

namespace ILines
{
	/**
	 * @brief Vertex program for the Phong/Blinn lighting model.
	 *
	 * This vertex program computes for each vertex the viewing vector,
	 * the light vector and the normalized tangent vector passing them to
	 * the fragment program in IL_cylinder_blinn_fp.cpp.
	 */
	const char *IL_cylinder_blinn_tangent_vp =
		"!!ARBvp1.0                                             \n"
		"                                                       \n"
		"# ******* attributes *********                         \n"
		"                                                       \n"
		"ATTRIB ipos  = vertex.position;                        \n"
		"ATTRIB iprev = vertex.texcoord[0];                     \n"
		"ATTRIB inext = vertex.texcoord[1];                     \n"
		"                                                       \n"
		"# ******* parameters *********                         \n"
		"                                                       \n"
		"PARAM mv_mat[]   = { state.matrix.modelview };         \n"
		"PARAM mvp_mat[]  = { state.matrix.mvp };               \n"
		"PARAM light_pos  = state.light[0].position;            \n"
		"PARAM const      = { 0, 0.5, 1, 2 };                   \n"
		"                                                       \n"
		"# ******** temporaries *********                       \n"
		"                                                       \n"
		"TEMP tangent;                                          \n"
		"TEMP eye_pos;                                          \n"
		"                                                       \n"
		"# ******* outputs *********                            \n"
		"                                                       \n"
		"OUTPUT opos     = result.position;                     \n"
		"OUTPUT otangent = result.texcoord[0];                  \n"
		"OUTPUT olight   = result.texcoord[1];                  \n"
		"OUTPUT oview    = result.texcoord[2];                  \n"
		"                                                       \n"
		"# ******** program *********                           \n"
		"                                                       \n"
		"# transform vertex position to camera coordinates      \n"
		"DP4 eye_pos.x, mv_mat[0], ipos;                        \n"
		"DP4 eye_pos.y, mv_mat[1], ipos;                        \n"
		"DP4 eye_pos.z, mv_mat[2], ipos;                        \n"
		"DP4 eye_pos.w, mv_mat[3], ipos;                        \n"
		"                                                       \n"
		"# compute tangent                                      \n"
		"SUB tangent, inext, iprev;                             \n"
		"DP3 tangent.w, tangent, tangent;                       \n"
		"RSQ tangent.w, tangent.w;                              \n"
		"MUL tangent.xyz, tangent, tangent.w;                   \n"
		"                                                       \n"
		"DP3 otangent.x, mv_mat[0], tangent;                    \n"
		"DP3 otangent.y, mv_mat[1], tangent;                    \n"
		"DP3 otangent.z, mv_mat[2], tangent;                    \n"
		"MOV otangent.w, const.z;                               \n"
		"                                                       \n"
		"# compute light direction                              \n"
		"SUB olight.xyz, light_pos, eye_pos;                    \n"
		"MOV olight.w, const.z;                                 \n"
		"                                                       \n"
		"# compute viewing direction                            \n"
		"MOV oview, -eye_pos;                                   \n"
		"                                                       \n"
		"# transform vertex position                            \n"
		"DP4 opos.x, mvp_mat[0], ipos;                          \n"
		"DP4 opos.y, mvp_mat[1], ipos;                          \n"
		"DP4 opos.z, mvp_mat[2], ipos;                          \n"
		"DP4 opos.w, mvp_mat[3], ipos;                          \n"
		"                                                       \n"
		"# pass through color                                   \n"
		"MOV result.color, vertex.color;                        \n"
		"                                                       \n"
		"END                                                    \n";
}

