#include "GLGPUTracer.h"
#include "io/GLGPUDataset.h"

GLGPUFieldLineTracer::GLGPUFieldLineTracer()
{
}

GLGPUFieldLineTracer::~GLGPUFieldLineTracer()
{
}

void GLGPUFieldLineTracer::SetDataset(const GLDataset *ds)
{
  _ds = (const GLGPUDataset*)ds;
}

void GLGPUFieldLineTracer::Trace()
{
  fprintf(stderr, "Trace..\n");
}
