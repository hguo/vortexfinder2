#ifndef _STORYLINE_WIDGET_H
#define _STORYLINE_WIDGET_H

#include <QGLWidget>
#include "common/VortexEvent.h"

class CStorylineWidget : public QGLWidget
{
  Q_OBJECT

public:
  CStorylineWidget(const QGLFormat& fmt=QGLFormat::defaultFormat(), QWidget *parent=NULL, QGLWidget *sharedWidget=NULL);
  ~CStorylineWidget();

protected:
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();
};

#endif
