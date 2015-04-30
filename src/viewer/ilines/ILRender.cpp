/* $Id: ILRender.cpp,v 1.67 2005/10/19 10:52:40 ovidiom Exp $ */

#include "ILRender.h"


namespace ILines
{
	struct ILRender::ILInfo
	{
		/** @brief Array of the starting element of each line. */
		GLint	*first;

		/** @brief Array of number of vertices of each line. */
		GLsizei	*vertCount;

		/** @brief Number of lines. */
		GLsizei	lineCount;

		/** @brief The vertices of the lines in homogeneous format. */
		float	*homVertices;

		/** @brief The tangents at each vertex of the lines. */
		float	*tangents;

		/** @brief VBO identifier for storing the tangents. */
		GLuint	tangentsVBO;
	};


	ILRender::ILRender()
	{
		isInitialized = false;
		doZSort = false;
		blinnVertProg = ShaderProgram(GL_VERTEX_PROGRAM_ARB);
		blinnTangentVertProg = ShaderProgram(GL_VERTEX_PROGRAM_ARB);
		blinnFragProg = ShaderProgram(GL_FRAGMENT_PROGRAM_ARB);
		errorCallback = NULL;
		lastError = IL_NO_ERROR;
		lastGLError = GL_NO_ERROR;
	}

	ILRender::~ILRender()
	{
		releaseResources();
	}

	/**
	 * Computes and sets up lighting textures for the given lighting model
	 * if the necessary extensions are available. An OpenGL rendering context
	 * must have been set up before calling this function. \n
	 * For a list of needed OpenGL extensions for the different lighting models,
	 * refer to the isLightingModelSupported(ILLightingModel::Model) function. \n
	 * <b>Errors:</b>
	 * - ILRender::IL_GL_ERROR is generated if an OpenGL error.
	 * - ILRender::IL_NO_ARB_MULTITEXTURE_EXTENSION is generated if the
	 *   GL_ARB_multitexture extension is needed but not supported.
	 * - ILRender::IL_NO_ARB_VERTEX_PROGRAM_EXTENSION is generated if the
	 *   GL_ARB_vertex_program extension is needed but not supported.
	 * - ILRender::IL_NO_ARB_fragment_PROGRAM_EXTENSION is generated if the
	 *   GL_ARB_fragment_program extension is needed but not supported.
	 * - ILRender::IL_INVALID_LIGHTING_MODEL is generated if an invalid lighting
	 *   model is supplied.
	 *
	 * @param ka             The ambient reflection coefficient.
	 * @param kd             The diffuse reflection coefficient.
	 * @param ks             The specular reflection coefficient.
	 * @param n              The gloss exponent for the specular lighting.
	 * @param texDim         The dimension of the \e square matrices.
	 * @param lightingModel  The lighting model to use.
	 * @param stretch        Flag whether to stretch the dynamic range.
	 * @param L              The \directional light vector in camera coordinates. \n
	 *                       Only required for the ILLightingModel::IL_MAXIMUM_PHONG
	 *                       and ILLightingModel::IL_CYLINDER_PHONG
	 *                       lighting models.
	 */
	void ILRender::setupTextures(float ka, float kd, float ks, float n, int texDim,
	                             ILLightingModel::Model lightingModel,
	                             bool stretch, const float *L)
	{
		float		*texDiff, *texSpec;
		/* The viewing vector in camera coordinates. */
		float		V[3] = { 0.0f, 0.0f, 1.0f };

		this->lightingModel = lightingModel;

		getExtensions();

		/* Clear a possible previous error. */
		getError();
		if (catchLightingModelErrors() != IL_NO_ERROR)
			return;

		saveRenderingState();

		releaseResources();

		texDiff = new float[texDim * texDim];
		texSpec = new float[texDim * texDim];

		switch (lightingModel)
		{
		case ILLightingModel::IL_MAXIMUM_PHONG:
		case ILLightingModel::IL_CYLINDER_PHONG:
			ILTexture::computeTextures(ka, kd, ks, n, texDim, NULL, texDiff, texSpec,
			                           lightingModel, stretch,
			                           normalize(L), normalize(V));

			buildTextureMatrix(normalize(L), normalize(V));
			break;
		case ILLightingModel::IL_CYLINDER_BLINN:
			ILTexture::computeTextures(ka, kd, ks, n, texDim, NULL, texDiff, texSpec,
			                           lightingModel, stretch);

			this->gloss = n;
			blinnVertProg.loadProgram(IL_cylinder_blinn_vp);
			blinnTangentVertProg.loadProgram(IL_cylinder_blinn_tangent_vp);
			blinnFragProg.loadProgram(IL_cylinder_blinn_fp);
			break;
		}

		initTexture(&texIDDiff, texDim, texDiff);
		initTexture(&texIDSpec, texDim, texSpec);

		/* The texture arrays are not needed anymore. */
		delete[] texDiff;
		delete[] texSpec;

		isInitialized = true;

		restoreRenderingState();

		catchGLErrors();
	}

	/**
	 * @param L  The \e normalized light vector.
	 * @param V  The \e normalized viewing vector.
	 */
	void ILRender::buildTextureMatrix(Vector3f L, Vector3f V)
	{
		// row 1
		textureMatrix[ 0] = 0.5f * L.x;
		textureMatrix[ 4] = 0.5f * L.y;
		textureMatrix[ 8] = 0.5f * L.z;
		textureMatrix[12] = 0.5f * 1.0f;

		// row 2
		textureMatrix[ 1] = 0.5f * V.x;
		textureMatrix[ 5] = 0.5f * V.y;
		textureMatrix[ 9] = 0.5f * V.z;
		textureMatrix[13] = 0.5f * 1.0f;

		// row 3
		textureMatrix[ 2] = 0.0f;
		textureMatrix[ 6] = 0.0f;
		textureMatrix[10] = 0.0f;
		textureMatrix[14] = 0.0f;

		// row 4
		textureMatrix[ 3] = 0.0f;
		textureMatrix[ 7] = 0.0f;
		textureMatrix[11] = 0.0f;
		textureMatrix[15] = 1.0f;
	}

	/**
	 * This function basically behaves like \a glMultiDrawArrays(). \n
	 * It renders a set of \a lineCount line strips starting at \a first[i]
	 * and consisting of \a vertCount[i] vertices. \n
	 * Note that \e no vertices should be shared among different lines
	 * since this implies that also texture coordinates used internally
	 * would then be erroneously shared. \n
	 * <b>Errors:</b>
	 * - ILRender::IL_NOT_INITIALIZED is generated if no lighting textures
	 *   have been set up before calling this function.
	 * - ILRender::IL_GL_ERROR is generated if an OpenGL error occurred.
	 *
	 * @param first      Array of indices into the OpenGL vertex array
	 *                   where the vertex set for each line strip starts.
	 * @param vertCount  Array of number of vertices forming each line strip.
	 * @param lineCount  Number of line strips to draw.
	 */
	void ILRender::multiDrawArrays(GLint *first,
	                               GLsizei *vertCount, GLsizei lineCount)
	{
		GLsizei	arrSize;
		GLvoid	*vertices;
		float	*homVertices, *tangents;
		bool	doNeedTangents;
		GLuint	mappedVBO;

		/* Clear a possible previous error. */
		getError();

		if (!isInitialized)
		{
			setError(IL_NOT_INITIALIZED);
			return;
		}

		saveRenderingState();

		vertices = getVertices(mappedVBO);
    
		/* Check whether there is anything to draw. */
		if (!glIsEnabled(GL_VERTEX_ARRAY) || (vertices == NULL))
		{
			releaseVertices(mappedVBO);
			restoreRenderingState();
			return;
		}

		arrSize = 0;
		for (int i = 0; i < lineCount; i++)
			if (first[i] + vertCount[i] > arrSize)
				arrSize = first[i] + vertCount[i];

		homVertices = NULL;
		tangents = NULL;

		doNeedTangents = (lightingModel != ILLightingModel::IL_CYLINDER_BLINN);

		/* The vertices are only needed if we have to compute the tangents
		 * or perform z-sorting. */
		if (doNeedTangents || doZSort)
			homVertices = getHomogeneous((int *)first, (int *)vertCount,
			                             (int)lineCount, arrSize);

		if (doNeedTangents)
			tangents = computeTangents((int *)first, (int *)vertCount,
			                           (int)lineCount, arrSize, homVertices);

		releaseVertices(mappedVBO);

		render((int *)first, (int *)vertCount, (int)lineCount,
		       homVertices, tangents);

		delete[] homVertices;
		delete[] tangents;

		restoreRenderingState();

		catchGLErrors();
	}

	/**
	 * This \e overloaded function does the same as the version above, but
	 * the parameters used are those previously passed to the
	 * \a prepareMultiDrawArrays() function which returned the here given
	 * identifier. \n
	 * Note that \a prepareMultiDrawArrays() does \e not save any rendering
	 * state, i.e. you still have to setup the whole rendering state,
	 * including all vertex arrays, as you would do with \a glMultiDrawArrays(). \n
	 * <b>Errors:</b>
	 * - ILRender::IL_NOT_INITIALIZED is generated if no lighting textures
	 *   have been set up before calling this function.
	 * - ILRender::IL_GL_ERROR is generated if an OpenGL error occurred.
	 *
	 * @param ilID  An identifier previously returned by
	 *              \a prepareMultiDrawArrays().
	 */
	void ILRender::multiDrawArrays(ILRender::ILIdentifier ilID)
	{
		ILInfo	*ilInfo;

		/* Clear a possible previous error. */
		getError();

		if (!isInitialized)
		{
			setError(IL_NOT_INITIALIZED);
			return;
		}

		if (ilID == IL_INVALID_IDENTIFIER)
			return;

		saveRenderingState();

		ilInfo = (ILInfo *)ilID;

		render((int *)ilInfo->first, (int *)ilInfo->vertCount,
		       (int)ilInfo->lineCount,
		       ilInfo->homVertices, ilInfo->tangents, ilInfo->tangentsVBO);

		restoreRenderingState();

		catchGLErrors();
	}

	/**
	 * This function takes the same parameters as \a multiDrawArrays()
	 * and preprocesses them for a more efficient rendering. The returned
	 * identifier can later be passed to \a multiDrawArrays(ILIdentifier)
	 * for the actual rendering. \n
	 * Prior to calling this function, at least the vertex coordinates
	 * must have been passed to OpenGL in a vertex array. \n
	 * Note that this function will \e not remember any rendering state. Thus,
	 * the rendering state still needs to be setup when calling
	 * \a multiDrawArrays(ILRender::ILIdentifier).
	 *
	 * @param first      Array of indices into the OpenGL vertex array
	 *                   where the vertex set for each line strip starts.
	 * @param vertCount  Array of number of vertices forming each line strip.
	 * @param lineCount  Number of line strips to draw.
	 * @return           A unique identifier for rendering the lines at a
	 *                   later time or ILRender::IL_INVALID_IDENTIFIER if some
	 *                   error occurred.
	 */
	ILRender::ILIdentifier ILRender::prepareMultiDrawArrays(GLint *first,
	                                                        GLsizei *vertCount,
	                                                        GLsizei lineCount)
	{
		GLvoid	*vertices;
		GLsizei	arrSize;
		ILInfo	*ilInfo;
		GLuint	mappedVBO, bindedVBO;

		getExtensions();

		/* Remember the currently binded VBO. */
		if (extVertexBufferObject)
			glGetIntegerv(GL_VERTEX_ARRAY_BUFFER_BINDING, (GLint *)&bindedVBO);

		vertices = getVertices(mappedVBO);

		/* Check whether there is anything to draw. */
		if (!glIsEnabled(GL_VERTEX_ARRAY) || (vertices == NULL))
		{
			releaseVertices(mappedVBO);
			/* Restore the previously binded VBO. */
			if (extVertexBufferObject)
				glBindBuffer(GL_ARRAY_BUFFER, bindedVBO);

			return (IL_INVALID_IDENTIFIER);
		}

		ilInfo = new ILInfo();

		arrSize = 0;
		for (int i = 0; i < lineCount; i++)
			if (first[i] + vertCount[i] > arrSize)
				arrSize = first[i] + vertCount[i];

		ilInfo->first = new GLint[lineCount];
		std::copy(first, first + lineCount, ilInfo->first);
		ilInfo->vertCount = new GLint[lineCount];
		std::copy(vertCount, vertCount + lineCount, ilInfo->vertCount);
		ilInfo->lineCount = lineCount;

		ilInfo->homVertices = getHomogeneous((int *)first, (int *)vertCount,
		                                     (int)lineCount, arrSize);

		ilInfo->tangents = computeTangents((int *)first, (int *)vertCount,
		                                   (int)lineCount, arrSize,
		                                   ilInfo->homVertices);

		releaseVertices(mappedVBO);

		if (!extVertexBufferObject)
			ilInfo->tangentsVBO = 0;
		else
		{
			glGenBuffers(1, &ilInfo->tangentsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, ilInfo->tangentsVBO);
			glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(ilInfo->tangents[0]) * arrSize,
			              ilInfo->tangents, GL_STATIC_DRAW);
		}

		/* Restore the previously binded VBO. */
		if (extVertexBufferObject)
			glBindBuffer(GL_ARRAY_BUFFER, bindedVBO);

		return (ilInfo);
	}

	/**
	 * @param ilID  An identifier previously returned by
	 *              \a prepareMultiDrawArrays().
	 */
	void ILRender::deleteIdentifier(ILRender::ILIdentifier ilID)
	{
		ILInfo	*ilInfo;

		if (ilID == IL_INVALID_IDENTIFIER)
			return;

		getExtensions();

		ilInfo = (ILInfo *)ilID;

		delete[] ilInfo->first;
		delete[] ilInfo->vertCount;
		delete[] ilInfo->homVertices;
		delete[] ilInfo->tangents;

		if ((ilInfo->tangentsVBO != 0) && extVertexBufferObject)
			glDeleteBuffers(1, &ilInfo->tangentsVBO);

		delete ilInfo;
	}

	/**
	 * @param size       The number of components of the texture coordinates.
	 * @param type       The type of the texture coordinates.
	 * @param stride     The stride between subsequent texture coordinates.
	 * @param texCoords  The texture coordinates.
	 */
	void ILRender::setupTexCoordArrays(GLint size, GLenum type, GLsizei stride,
	                                   const GLvoid *texCoords) const
	{
		float	*rotationMatrix;

		switch (lightingModel)
		{
		case ILLightingModel::IL_MAXIMUM_PHONG:
		case ILLightingModel::IL_CYLINDER_PHONG:
			rotationMatrix = ILUtilities::computeRotationMatrix();

			glMatrixMode(GL_TEXTURE);
			glLoadMatrixf(textureMatrix);
			glMultMatrixf(rotationMatrix);
			break;
		case ILLightingModel::IL_CYLINDER_BLINN:
			glMatrixMode(GL_TEXTURE);
			glLoadIdentity();
			break;
		}

		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(size, type, stride, texCoords);
	}

	/**
	 * @param textureMode  The texture mode to be enabled.
	 */
	void ILRender::enableTexMode(GLenum textureMode)
	{
		glDisable(GL_TEXTURE_1D);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_TEXTURE_3D);

		glEnable(textureMode);
	}

	/**
	 * @param size           The number of components of the texture coordinates.
	 * @param type           The type of the texture coordinates.
	 * @param stride         The stride between subsequent texture coordinates.
	 * @param texCoordsDiff  The texture coordinates for the diffuse lighting.
	 * @param texCoordsSpec  The texture coordinates for the specular lighting.
	 * @param texCoordsVBO   The identifier of the vertex buffer object containing
	 *                       the texture coordinates or 0 if not available.
	 */
	void ILRender::setupTexUnits(GLint size, GLenum type, GLsizei stride,
	                             const GLvoid *texCoordsDiff, const GLvoid *texCoordsSpec,
	                             GLuint texCoordsVBO) const
	{
		if (extVertexBufferObject)
			glBindBuffer(GL_ARRAY_BUFFER, texCoordsVBO);

		glActiveTexture(GL_TEXTURE0);
		enableTexMode(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texIDDiff);
		glClientActiveTexture(GL_TEXTURE0);
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		setupTexCoordArrays(size, type, stride, texCoordsDiff);

		glActiveTexture(GL_TEXTURE1);
		enableTexMode(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texIDSpec);
		glClientActiveTexture(GL_TEXTURE1);
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_ADD);
		setupTexCoordArrays(size, type, stride, texCoordsSpec);

		if (extVertexBufferObject)
			glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	/**
	 * @return  The vertex stride.
	 */
	GLsizei ILRender::getVertexStride()
	{
		GLenum	type;
		GLint	size;
		GLsizei	stride;

		glGetIntegerv(GL_VERTEX_ARRAY_STRIDE, (GLint *)&stride);

		if (stride != 0)
			return (stride);

		glGetIntegerv(GL_VERTEX_ARRAY_TYPE, (GLint *)&type);
		glGetIntegerv(GL_VERTEX_ARRAY_SIZE, (GLint *)&size);

		switch (type)
		{
		case GL_SHORT:
			return (size * sizeof(GLshort));
		case GL_INT:
			return (size * sizeof(GLint));
		case GL_FLOAT:
			return (size * sizeof(GLfloat));
		case GL_DOUBLE:
			return (size * sizeof(GLdouble));
		default:
			return (0);
		}
	}

	/**
	 * @param first        Array of indices into the OpenGL vertex array
	 *                     where the vertex set for each line strip starts.
	 * @param vertCount    Array of number of vertices forming each line strip.
	 * @param lineCount    Number of line strips to draw.
	 * @param homVertices  The \e homogeneous vertex coordinates.
	 * @param tangents     The tangent vectors at the vertices.
	 * @param tangentsVBO  The identifier of the vertex buffer object containing
	 *                     the tangents or 0 if not available.
	 */
	void ILRender::render(int *first, int *vertCount, int lineCount,
	                      const float *homVertices,
	                      const float *tangents, GLuint tangentsVBO) const
	{
		int		i;
		GLvoid	*vertices;
		GLint	size;
		GLenum	type;
		GLsizei	stride;
		GLuint	vboID;

		switch (lightingModel)
		{
		case ILLightingModel::IL_MAXIMUM_PHONG:
		case ILLightingModel::IL_CYLINDER_PHONG:
			if (tangentsVBO == 0)
				setupTexUnits(3, GL_FLOAT, 0, tangents, tangents, 0);
			else
				setupTexUnits(3, GL_FLOAT, 0, 0, 0, tangentsVBO);

			renderLines(first, vertCount, lineCount, homVertices);
			break;
		case ILLightingModel::IL_CYLINDER_BLINN:
			glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, gloss / 2.0f);

			if (tangents != NULL)
			{
				blinnVertProg.bind();
				blinnFragProg.bind();

				if (tangentsVBO == 0)
					setupTexUnits(3, GL_FLOAT, 0, tangents, tangents, 0);
				else
					setupTexUnits(3, GL_FLOAT, 0, 0, 0, tangentsVBO);

				renderLines(first, vertCount, lineCount, homVertices);

				blinnVertProg.release();
				blinnFragProg.release();
			}
			else
			{
				blinnTangentVertProg.bind();
				blinnFragProg.bind();

				glGetIntegerv(GL_VERTEX_ARRAY_SIZE, &size);
				glGetIntegerv(GL_VERTEX_ARRAY_TYPE, (GLint *)&type);
				stride = getVertexStride();
				glGetPointerv(GL_VERTEX_ARRAY_POINTER, &vertices);
				vboID = 0;
				if (extVertexBufferObject)
					glGetIntegerv(GL_VERTEX_ARRAY_BUFFER_BINDING, (GLint *)&vboID);

				/* Render the first line segment of each line. */
				setupTexUnits(size, type, stride,
				              vertices, (char *)vertices + stride, vboID);
				for (i = 0; i < lineCount; i++)
					glDrawArrays(GL_LINES, first[i], 2);

				/* Render the last line segment of each line. */
				setupTexUnits(size, type, stride,
				              (char *)vertices - stride, vertices, vboID);
				for (i = 0; i < lineCount; i++)
					glDrawArrays(GL_LINES, first[i] + vertCount[i] - 2, 2);

				/* Render all other line segments. */
				setupTexUnits(size, type, stride,
				              (char *)vertices - stride, (char *)vertices + stride,
				              vboID);

				for (i = 0; i < lineCount; i++)
				{
					first[i]++;
					vertCount[i] -= 2;
				}

				renderLines(first, vertCount, lineCount, homVertices);

				for (i = 0; i < lineCount; i++)
				{
					first[i]--;
					vertCount[i] += 2;
				}

				blinnTangentVertProg.release();
				blinnFragProg.release();
			}
			break;
		default:
			return;
		}
	}

	/**
	 * @param first        Array of indices into the OpenGL vertex array
	 *                     where the vertex set for each line strip starts.
	 * @param vertCount    Array of number of vertices forming each line strip.
	 * @param lineCount    Number of line strips to draw.
	 * @param homVertices  The \e homogeneous vertex coordinates.
	 */
	void ILRender::renderLines(int *first, int *vertCount, int lineCount,
	                           const float *homVertices) const
	{
		int	i;
		int	*indices;
		int	segCount;

		if (!doZSort)
		{
			if (extMultiDrawArrays)
				glMultiDrawArrays(GL_LINE_STRIP, first, vertCount, lineCount);
			else
				for (i = 0; i < lineCount; i++)
					glDrawArrays(GL_LINE_STRIP, first[i], vertCount[i]);
		}
		else
		{
			indices = ILUtilities::zsort(first, vertCount, lineCount, homVertices);

			/* Compute the number of line segments. */
			segCount = 0;
			for (i = 0; i < lineCount; i++)
				segCount += vertCount[i] - 1;

			GLint vboVertexBinding;

			vboVertexBinding = 0;
			if (extVertexBufferObject)
				glGetIntegerv(GL_VERTEX_ARRAY_BUFFER_BINDING, &vboVertexBinding);

			if (vboVertexBinding == 0)
				for (i = 0; i < segCount; i++)
					glDrawArrays(GL_LINES, indices[i], 2);
			else
			{
				GLuint *idx = new GLuint[2 * segCount];
				for (i = 0; i < segCount; i++)
				{
					idx[2 * i + 0] = indices[i];
					idx[2 * i + 1] = indices[i] + 1;
				}

				glDrawElements(GL_LINES, 2 * segCount, GL_UNSIGNED_INT, idx);

				delete[] idx;
			}

			delete[] indices;
		}
	}

	void ILRender::saveRenderingState()
	{
		glGetIntegerv(GL_MATRIX_MODE, (GLint *)&stateMatrixMode);
		glGetIntegerv(GL_ACTIVE_TEXTURE, (GLint *)&stateActiveTexture);
		glGetIntegerv(GL_CLIENT_ACTIVE_TEXTURE, (GLint *)&stateClientActiveTexture);
		glGetLightfv(GL_LIGHT0, GL_SPOT_EXPONENT, &stateSpotExponent);

		glActiveTexture(GL_TEXTURE0);
		glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT);
		glGetDoublev(GL_TEXTURE_MATRIX, stateTextureMatrixDiff);

		glActiveTexture(GL_TEXTURE1);
		glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT);
		glGetDoublev(GL_TEXTURE_MATRIX, stateTextureMatrixSpec);

		/* Needed for the vertex buffer object bindings. */
		glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
	}

	void ILRender::restoreRenderingState()
	{
		glPopClientAttrib();

		glActiveTexture(GL_TEXTURE1);
		glPopAttrib();
		glMatrixMode(GL_TEXTURE);
		glLoadMatrixd(stateTextureMatrixSpec);

		glActiveTexture(GL_TEXTURE0);
		glPopAttrib();
		glMatrixMode(GL_TEXTURE);
		glLoadMatrixd(stateTextureMatrixDiff);

		glMatrixMode(stateMatrixMode);
		glActiveTexture(stateActiveTexture);
		glClientActiveTexture(stateClientActiveTexture);
		glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, stateSpotExponent);
	}


	/**
	 * @param extension  The name of the extension to be checked for.
	 * @return           Whether the given extension is supported or not.
	 */
	bool ILRender::isExtensionSupported(const char *extension)
	{
		char	*extString, *match;

		extString = (char *)glGetString(GL_EXTENSIONS);
		if (extString == NULL)
			return (false);

		match = strstr(extString, extension);
		if (match == NULL)
			return (false);

		if (match[strlen(extension)] != ' ')
			return (false);
		if ((match == extString) || (match[-1] == ' '))
			return (true);
		return (false);
	}

	/**
	 * @param texID    The address to the texture ID to be initialized.
	 * @param texDim   The dimension of the texture to be initialized.
	 * @param texture  The address to the texture data.
	 */
	void ILRender::initTexture(GLuint *texID, int texDim, const float *texture)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glGenTextures(1, texID);
		glBindTexture(GL_TEXTURE_2D, *texID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, texDim, texDim, 0,
		             GL_LUMINANCE, GL_FLOAT, texture);
	}

	/**
	 * @param first      Array of indices into the OpenGL vertex array
	 *                   where the vertex set for each line strip starts.
	 * @param vertCount  Array of number of vertices forming each line strip.
	 * @param lineCount  Number of line strips to draw.
	 * @param arrSize    The size of the array used for the vertices.
	 * @return           The \a vertices transformed to \e homogeneous
	 *                   coordinates.
	 */
	float *ILRender::getHomogeneous(int *first, int *vertCount, int lineCount,
	                                int arrSize)
	{
		GLvoid	*vertices;
		GLenum	vertType;
		GLint	vertSize;
		GLsizei	vertStride;
		float	*homVertices;
		GLuint	mappedVBO;

		vertices = getVertices(mappedVBO);

		glGetIntegerv(GL_VERTEX_ARRAY_TYPE, (GLint *)&vertType);
		glGetIntegerv(GL_VERTEX_ARRAY_SIZE, (GLint *)&vertSize);
		glGetIntegerv(GL_VERTEX_ARRAY_STRIDE, (GLint *)&vertStride);

		switch (vertType)
		{
		case GL_SHORT:
			homVertices = toHomogeneous(first, vertCount, lineCount,
			                            vertSize, vertStride,
			                            arrSize, (GLshort *)vertices);
			break;
		case GL_INT:
			homVertices = toHomogeneous(first, vertCount, lineCount,
			                            vertSize, vertStride,
			                            arrSize, (GLint *)vertices);
			break;
		case GL_FLOAT:
			homVertices = toHomogeneous(first, vertCount, lineCount,
			                            vertSize, vertStride,
			                            arrSize, (GLfloat *)vertices);
			break;
		case GL_DOUBLE:
			homVertices = toHomogeneous(first, vertCount, lineCount,
			                            vertSize, vertStride,
			                            arrSize, (GLdouble *)vertices);
			break;
		default:
			homVertices = NULL;
			break;
		}

		releaseVertices(mappedVBO);

		return (homVertices);
	}

	/**
	 * @param first      Array of indices into the OpenGL vertex array
	 *                   where the vertex set for each line strip starts.
	 * @param vertCount  Array of number of vertices forming each line strip.
	 * @param lineCount  Number of line strips to draw.
	 * @param size       Number of components (2, 3 or 4) provided for the
	 *                   coordinates of each vertex.
	 * @param stride     The byte offset between consecutive vertices.
	 *                   If stride is 0,  the  vertices are understood to be
	 *                   tightly packed in the array.
	 * @param arrSize    The size of the array used for the vertices.
	 * @param vertices   \e Template parameter. \n
	 *                   The vertex array passed to OpenGL by the user
	 *                   These vertices may be in any format accepted by OpenGL.
	 * @return           The \a vertices transformed to \e homogeneous
	 *                   coordinates.
	 */
	template <typename T>
	float *ILRender::toHomogeneous(int *first, int *vertCount, int lineCount,
	                               int size, int stride,
	                               int arrSize, const T *vertices)
	{
		int		idx;
		float	*homVertices;
		const T	*vertex;

		/* Treat the special case of stride being zero. */
		if (stride == 0)
			stride = size * sizeof(T);

		homVertices = new float[4 * arrSize];

		for (int i = 0; i < lineCount; i++)
			for (int j = 0; j < vertCount[i]; j++)
			{
				idx = first[i] + j;

				vertex = (const T *)((char *)vertices + stride * idx);

				homVertices[4 * idx + 0] = (float)vertex[0];
				homVertices[4 * idx + 1] = (float)vertex[1];
				homVertices[4 * idx + 2] = (size > 2) ? (float)vertex[2] : 0.0f;
				homVertices[4 * idx + 3] = (size > 3) ? (float)vertex[3] : 1.0f;
			}

		return (homVertices);
	}

	/**
	 * @param first      Array of indices into the OpenGL vertex array
	 *                   where the vertex set for each line strip starts.
	 * @param vertCount  Array of number of vertices forming each line strip.
	 * @param lineCount  Number of line strips to draw.
	 * @param arrSize    The size of the array used for the vertices.
	 * @param homVert    The vertices in \e homogeneous format.
	 * @return           The tangent vectors at the vertices.
	 */
	float *ILRender::computeTangents(int *first, int *vertCount, int lineCount,
	                                 int arrSize, const float *homVert)
	{
		float		*tangents;
		int			idx, idx1, idx2;
		Vector3f	vert1, vert2, tangent;

		tangents = new float[3 * arrSize];

		for (int i = 0; i < lineCount; i++)
			for (int j = 0; j < vertCount[i]; j++)
			{
				idx = first[i] + j;

				idx1 = (j > 0) ? idx - 1 : first[i];
				idx2 = (j < vertCount[i] - 1) ? idx + 1 : idx;

				vert1 = Vector3f(&homVert[4 * idx1]);
				vert2 = Vector3f(&homVert[4 * idx2]);

				tangent = normalize(vert2 - vert1);
				tangents[3 * idx + 0] = tangent.x;
				tangents[3 * idx + 1] = tangent.y;
				tangents[3 * idx + 2] = tangent.z;
			}

		return (tangents);
	}

	/**
	 * @return  The texture unit used for the ambient and diffuse lighting.
	 */
	GLuint ILRender::getTexIDDiff() const
	{
		return (texIDDiff);
	}

	/**
	 * @return  The texture unit used for the specular lighting.
	 */
	GLuint ILRender::getTexIDSpec() const
	{
		return (texIDSpec);
	}

	/**
	 * @param doZSort  Flag whether to enable z-sorting.
	 */
	void ILRender::enableZSort(bool doZSort)
	{
		this->doZSort = doZSort;
	}

	/**
	 * @return  Whether z-sorting is enabled.
	 */
	bool ILRender::isEnabledZSort() const
	{
		return (doZSort);
	}

	/**
	 * Checks whether the passed lighting model is supported by
	 * checking for the presence of the OpenGL extensions needed
	 * for the given lighting model. An OpenGL rendering context
	 * must have been set up before calling this function. \n
	 * The needed OpenGL extensions for the different lighting models are:
	 * - ILLightingModel::IL_MAXIMUM_PHONG:
	 *   - GL_ARB_multitexture
	 * - ILLightingModel::IL_CYLINDER_BLINN:
	 *   - GL_ARB_multitexture
	 *   - GL_ARB_vertex_program
	 *   - GL_ARB_fragment_program
	 * - ILLightingModel::IL_CYLINDER_PHONG:
	 *   - GL_ARB_multitexture
	 *
	 * @param model  The lighting model to be checked for.
	 * @return       Whether the passed lighting model is supported.
	 */
	bool ILRender::isLightingModelSupported(ILLightingModel::Model model)
	{
		switch (model)
		{
		case ILLightingModel::IL_MAXIMUM_PHONG:
		case ILLightingModel::IL_CYLINDER_PHONG:
			return (isExtensionSupported("GL_ARB_multitexture"));
		case ILLightingModel::IL_CYLINDER_BLINN:
			return (isExtensionSupported("GL_ARB_multitexture") &&
			        isExtensionSupported("GL_ARB_vertex_program") &&
			        isExtensionSupported("GL_ARB_fragment_program"));
		default:
			return (false);
		}
	}

	/**
	 * Returns a code representing the error generated by the last
	 * operation and resets the error flag to ILRender::IL_NO_ERROR. \n
	 * The functions which may generate an error are:
	 * - setupTextures()
	 * - multiDrawArrays(GLint *, GLsizei *, GLsizei)
	 * - multiDrawArrays(ILIdentifier)
	 *
	 * @return  The error generated by the last operation.
	 */
	ILRender::ILError ILRender::getError()
	{
		ILError	err;

		err = lastError;
		lastError = IL_NO_ERROR;

		return (err);
	}

	/**
	 * Returns a code representing the OpenGL error generated by the last
	 * operation. This error is only meaningful, if a ILRender::IL_GL_ERROR
	 * has been generated.
	 *
	 * @return  The OpenGL error generated by the last operation.
	 */
	GLenum ILRender::getGLError()
	{
		GLenum	err;

		err = lastGLError;
		lastGLError = GL_NO_ERROR;

		return (err);
	}

	/**
	 * Records the occurred error which can later be queried through the getError()
	 * function and invokes the error function previously set through
	 * setErrorCallback(ILRender::ILErrorCallback), if any.
	 *
	 * @param err  The error which occurred.
	 */
	void ILRender::setError(ILRender::ILError err)
	{
		lastError = err;

		if (errorCallback != NULL)
			errorCallback(this);
	}

	/**
	 * @param err  The error to get the description for.
	 * @return     The description of the error or NULL if an invalid
	 *             error code is passed.
	 */
	const char *ILRender::errorString(ILRender::ILError err)
	{
		switch (err)
		{
		case IL_NO_ERROR:
			return ("No error occurred.");
		case IL_NOT_INITIALIZED:
			return ("Lighting textures have not been set up before rendering.");
		case IL_GL_ERROR:
			return ("An OpenGL error occurred.");
		case IL_INVALID_LIGHTING_MODEL:
			return ("Support for an invalid lighting model has been requested.");
		case IL_NO_ARB_MULTITEXTURE_EXTENSION:
			return ("GL_ARB_multitexture extension is not supported.");
		case IL_NO_EXT_MULTI_DRAW_ARRAYS_EXTENSION:
			return ("GL_EXT_multi_draw_arrays extension is not supported.");
		case IL_NO_ARB_VERTEX_PROGRAM_EXTENSION:
			return ("GL_ARB_vertex_program extension is not supported.");
		case IL_NO_ARB_FRAGMENT_PROGRAM_EXTENSION:
			return ("GL_ARB_fragment_program extension is not supported.");
		default:
			return (NULL);
		}
	}

	/**
	 * @return  The currently used error callback function or NULL, if none is used.
	 */
	ILRender::ILErrorCallback ILRender::getErrorCallback() const
	{
		return (errorCallback);
	}

	/**
	 * @param errorCallback  The error callback function to be used or NULL to
	 *                       disable the error callback mechanism.
	 */
	void ILRender::setErrorCallback(ILRender::ILErrorCallback errorCallback)
	{
		this->errorCallback = errorCallback;
	}

	/**
	 * Checks that the necessary OpenGL extensions are present to
	 * support the currently set lighting model. \n
	 * For a list of needed OpenGL extensions for the different lighting models,
	 * refer to the isLightingModelSupported(ILLightingModel::Model) function.
	 *
	 * @return  The catched error.
	 */
	ILRender::ILError ILRender::catchLightingModelErrors()
	{
		switch (lightingModel)
		{
		case ILLightingModel::IL_MAXIMUM_PHONG:
		case ILLightingModel::IL_CYLINDER_PHONG:
			if (!isExtensionSupported("GL_ARB_multitexture"))
				setError(IL_NO_ARB_MULTITEXTURE_EXTENSION);
			break;
		case ILLightingModel::IL_CYLINDER_BLINN:
			if (!isExtensionSupported("GL_ARB_multitexture"))
				setError(IL_NO_ARB_MULTITEXTURE_EXTENSION);
			if (!isExtensionSupported("GL_ARB_vertex_program"))
				setError(IL_NO_ARB_VERTEX_PROGRAM_EXTENSION);
			if (!isExtensionSupported("GL_ARB_fragment_program"))
				setError(IL_NO_ARB_FRAGMENT_PROGRAM_EXTENSION);
			break;
		default:
			setError(IL_INVALID_LIGHTING_MODEL);
			break;
		}

		return (lastError);
	}

	/**
	 * Checks whether an OpenGL error occurred in which case the error flag
	 * is set to ILRender::IL_GL_ERROR.
	 */
	void ILRender::catchGLErrors()
	{
		lastGLError = glGetError();
		if (lastGLError != GL_NO_ERROR)
			setError(IL_GL_ERROR);
	}

	/**
	 * If the vertices passed to OpenGL are stored in a vertex buffer object
	 * which is not mapped, the vertex buffer object is mapped and its
	 * identifier is stored in \a mappedVBO.
	 *
	 * @param mappedVBO  The identifier of the mapped vertex buffer object or
	 *                   0 if no vertex buffer object has been mapped.
	 * @return           Pointer to the vertices supplied to OpenGL by the user.
	 */
	GLvoid *ILRender::getVertices(GLuint &mappedVBO)
	{
		GLvoid	*vertices, *vboMapped;
		GLuint	verticesVBO;
		long	offset;

		mappedVBO = 0;

		glGetPointerv(GL_VERTEX_ARRAY_POINTER, &vertices);

		verticesVBO = 0;
		if (extVertexBufferObject)
			glGetIntegerv(GL_VERTEX_ARRAY_BUFFER_BINDING, (GLint *)&verticesVBO);

		/* If no vertex buffer object is binded, return the vertex array pointer. */
		if ((verticesVBO == 0) || !extVertexBufferObject)
			return (vertices);

		/* If vertex buffer objects are used, the pointers represent offsets
		 * in basic machine units. */
		offset = (long)vertices;

		glBindBuffer(GL_ARRAY_BUFFER, verticesVBO);

		/* Check whether the vertex buffer object data is already mapped. */
		glGetBufferPointerv(GL_ARRAY_BUFFER, GL_BUFFER_MAP_POINTER, &vboMapped);
		if (vboMapped != NULL)
			return ((char *)vboMapped + offset);

		/* Map the vertex buffer object data and return it. */
		vboMapped = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

		mappedVBO = verticesVBO;

		return ((char *)vboMapped + offset);
	}

	/**
	 * If a vertex buffer object has been mapped during the last call to
	 * getVertices(GLuint&), it will be unmapped here.
	 *
	 * @param mappedVBO  The identifier of the mapped vertex buffer object or
	 *                   0 if no vertex buffer object has been mapped.
	 */
	void ILRender::releaseVertices(GLuint mappedVBO)
	{
		if ((mappedVBO != 0) && extVertexBufferObject)
		{
			glBindBuffer(GL_ARRAY_BUFFER, mappedVBO);
			glUnmapBuffer(GL_ARRAY_BUFFER);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	}

	void ILRender::releaseResources()
	{
		if (texIDDiff != 0)
			glDeleteTextures(1, &texIDDiff);
		if (texIDSpec != 0)
			glDeleteTextures(1, &texIDSpec);

		texIDDiff = 0;
		texIDSpec = 0;
	}


	const ILRender::ILIdentifier	ILRender::IL_INVALID_IDENTIFIER   = NULL;
	
	bool							ILRender::extMultitexture         = false;
	bool							ILRender::extMultiDrawArrays      = false;
	bool							ILRender::extVertexBufferObject   = false;
	bool							ILRender::extVertexProgram        = false;
	bool							ILRender::extFragmentProgram      = false;
}

