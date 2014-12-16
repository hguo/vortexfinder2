#include "Tracer.h"

FieldLineTracer::FieldLineTracer()
{

}

FieldLineTracer::~FieldLineTracer()
{

}

void FieldLineTracer::WriteFieldLines(const std::string& filename)
{
  ::WriteFieldLines(filename, _fieldlines);
}
