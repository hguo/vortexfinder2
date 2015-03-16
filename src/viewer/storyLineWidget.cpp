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

CStorylineWidget::CStorylineWidget(const QGLFormat& fmt, QWidget *parent, QGLWidget *sharedWidget) :
  _vmap(NULL)
{

}

CStorylineWidget::~CStorylineWidget()
{

}

static bool conflict(int ts0, int tl0, int ts, int tl, int gap = 2)
{
  if (ts + tl + gap <= ts0) return true;
  else if (ts >= ts0 + tl0 + gap) return true;
  return false;
}

void CStorylineWidget::SetSequenceMap(const VortexSequenceMap *vmap)
{
  _vmap = vmap;

  // pass 1
  for (int i=0; i<_vmap->size(); i++) {
    bool fit = true;
    int fit_slot = -1;
    for (int j=0; j<_slots.size(); j++) {
      for (std::set<int>::iterator it = _slots[j].begin(); it != _slots[j].end(); it ++) {
        int i0 = *it;
        if (conflict(vmap->at(i0).ts, vmap->at(i0).tl, vmap->at(i).ts, vmap->at(i).tl), 10) {
          fit = false;
          break;
        }
      }
      if (fit) {
        fit_slot = j;
        break;
      }
    }

    if (fit_slot<0) fit = false;
    // fprintf(stderr, "fit=%d, slot=%d\n", fit, fit_slot);

next:
    if (fit) {
      _slots[fit_slot].insert(i);
    } else {
      std::set<int> slot;
      slot.insert(i);
      _slots.push_back(slot);
    }
  }

  fprintf(stderr, "nlines=%d, nslots=%d\n", vmap->size(), _slots.size());
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
  const int margin = 10; // 10px
  _rect_chart = QRectF(margin, margin, w-2*margin, h-2*margin);
}

void CStorylineWidget::renderRect()
{
  glColor3f(0, 0, 0);
  glBegin(GL_LINE_LOOP);
  glVertex2f(_rect_chart.left(), _rect_chart.bottom());
  glVertex2f(_rect_chart.right(), _rect_chart.bottom());
  glVertex2f(_rect_chart.right(), _rect_chart.top());
  glVertex2f(_rect_chart.left(), _rect_chart.top());
  glEnd();
}

void CStorylineWidget::renderLines()
{
  const int nlines = _vmap->size();

  for (int i=0; i<_slots.size(); i++) {
    for (std::set<int>::iterator it = _slots[i].begin(); it != _slots[i].end(); it ++) {
      const int k = *it;
      // fprintf(stderr, "slot=%d, id=%d\n", i, k);

      const float x0 = _rect_chart.x() + (_vmap->at(k).ts - _vmap->ts())  *_rect_chart.width()/_vmap->tl();
      const float x1 = _rect_chart.x() + (_vmap->at(k).ts + _vmap->at(k).tl - _vmap->ts())*_rect_chart.width()/_vmap->tl();
      const float y = _rect_chart.y() + k*_rect_chart.height()/nlines;

      glColor3f(0, 0, 0);
      glBegin(GL_LINES);
      glVertex2f(x0, y);
      glVertex2f(x1, y);
      glEnd();
    }
  }
}

void CStorylineWidget::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, width(), height());

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, width(), 0, height());
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // renderRect();
  renderLines();

  CHECK_GLERROR();
}
