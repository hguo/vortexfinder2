#ifndef _STORYLINE_WIDGET_H
#define _STORYLINE_WIDGET_H

#include <QGLWidget>
#include <QRect>
#include <QVector2D>
#include <set>
#include "common/VortexTransition.h"

class CStorylineWidget : public QGLWidget
{
  Q_OBJECT

public:
  CStorylineWidget(const QGLFormat& fmt=QGLFormat::defaultFormat(), QWidget *parent=NULL, QGLWidget *sharedWidget=NULL);
  ~CStorylineWidget();

  void SetVortexTrasition(const VortexTransition *vt);

protected:
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();

  void keyPressEvent(QKeyEvent *e);

protected:
  void renderLines();
  void renderRect();

private:
  void parseLayout();
  void saveEps();

private:
  QRectF _rect_chart;
  const VortexTransition *_vt;

  QMap<QPair<int, int>, QVector2D> _coords;
  float _layout_width, _layout_height;
};

#endif
