/* $Id: ShaderProgram.cpp,v 1.8 2005/10/17 10:12:09 ovidiom Exp $ */

#include "ShaderProgram.h"


namespace ILines
{
	ShaderProgram::ShaderProgram(GLenum target)
	{
		this->target = target;
		this->programID = 0;
	}

	ShaderProgram::~ShaderProgram()
	{
		destroy();
	}

	void ShaderProgram::bind() const
	{
		getExtensions();

		if (programID == 0)
			const_cast<ShaderProgram *>(this)->allocate();

		glEnable(target);
		glBindProgramARB(target, programID);
	}

	void ShaderProgram::release() const
	{
		if (programID != 0)
			glDisable(target);
	}

	void ShaderProgram::allocate()
	{
		getExtensions();

		destroy();
		glGenProgramsARB(1, &programID);
	}

	void ShaderProgram::destroy()
	{
		getExtensions();

		if (programID != 0)
		{
			glDeleteProgramsARB(1, &programID);
			programID = 0;
		}
	}

	bool ShaderProgram::load(std::istream &is) const
	{
		std::string	line, assembly;

		getExtensions();

		/* Load the fragment program. */
		do
		{
			std::getline(is, line);
			assembly.append(line);
			assembly.append("\n");
		} while (is);

		return (loadProgram(assembly));
	}

	bool ShaderProgram::load(const std::string &fn) const
	{
		getExtensions();

		std::ifstream f(fn.c_str());
		if (!f) 
		{ 
			std::cout << "Couldn't load arb program '" << fn << "'\n"; 
			return (false);
		}

#ifdef _DEBUG
		std::cout << "loading arb program from '" << fn << "'\n";
#endif

		f >> std::noskipws;

		return (load(f));
	}

	bool ShaderProgram::loadProgram(const std::string &program) const
	{
		bool	succ;

		bind();

		glProgramStringARB(target, GL_PROGRAM_FORMAT_ASCII_ARB, 
		                 GLsizei(program.length()), program.c_str());

		if (succ = (glGetError() != GL_INVALID_OPERATION))
		{
#if 0
#ifdef _DEBUG
			std::cout << "arb program loaded successfully.\n";
			int n;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_INSTRUCTIONS, &n);
			std::cout << "\tinstructions    \t" << n << std::endl;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS, &n);
			std::cout << "\talu instructions\t" << n << std::endl;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS, &n);
			std::cout << "\ttex instructions\t" << n << std::endl;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_TEX_INDIRECTIONS, &n);
			std::cout << "\ttex indirections\t" << n << std::endl;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_TEMPORARIES, &n);
			std::cout << "\ttemporaries     \t" << n << std::endl;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_PARAMETERS, &n);
			std::cout << "\tparameters      \t" << n << std::endl;
			glGetProgramivARB(target, GL_PROGRAM_NATIVE_ATTRIBS, &n);
			std::cout << "\tattributes      \t" << n << std::endl;
#endif
#endif
		}
		else
		{
			std::cerr << "Failed!" << std::endl;
#if 0
			GLint errPos;
			glGetIntegerv(GL_PROGRAM_ERROR_POSITION, &errPos);
			std::cerr << "arb program error at position " << errPos << ":\n";
			std::istringstream isstr((char*)glGetString(GL_PROGRAM_ERROR_STRING));
			std::string line;
			while(std::getline(isstr, line))
				std::cerr << "\t" << line << std::endl;
#endif
		}

		release();

		return (succ);
	}

	GLuint ShaderProgram::getProgramID() const
	{
		return (programID);
	}

	void ShaderProgram::setProgramID(GLuint programID)
	{
		this->programID = programID;
	}
}

