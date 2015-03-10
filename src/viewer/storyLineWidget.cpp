#include <QMouseEvent>
#include <QDebug>
#include "def.h"
#include "storyLineWidget.h"
#include "common/VortexTransition.h"
#include "common/Utils.hpp"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glu.h>
#include <GL/glut.h>
#endif

CStorylineWidget::CStorylineWidget(const QGLFormat& fmt, QWidget *parent, QGLWidget *sharedWidget)
{

}

CStorylineWidget::~CStorylineWidget()
{

}

void CStorylineWidget::initializeGL()
{
  glEnable(GL_MULTISAMPLE);

  GLint bufs, samples; 
  glGetIntegerv(GL_SAMPLE_BUFFERS, &bufs); 
  glGetIntegerv(GL_SAMPLES, &samples); 

  glEnable(GL_LINE_SMOOTH); 
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); 
  
  glEnable(GL_POLYGON_SMOOTH); 
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST); 

  glEnable(GL_BLEND); 
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

  glClearColor(1, 1, 1, 0);
}

void CStorylineWidget::resizeGL(int w, int h)
{
  glViewport(0, 0, w, h);
}

void CStorylineWidget::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, width(), 0, height());
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  CHECK_GLERROR();
}
