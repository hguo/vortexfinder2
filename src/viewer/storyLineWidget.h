#ifndef _STORYLINE_WIDGET_H
#define _STORYLINE_WIDGET_H

#include <QGLWidget>
#include <QRect>
#include <set>
#include "common/VortexSequence.h"

class CStorylineWidget : public QGLWidget
{
  Q_OBJECT

public:
  CStorylineWidget(const QGLFormat& fmt=QGLFormat::defaultFormat(), QWidget *parent=NULL, QGLWidget *sharedWidget=NULL);
  ~CStorylineWidget();

  void SetSequenceMap(const VortexSequenceMap *vmap);

protected:
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();

protected:
  void renderLines();
  void renderRect();

private:
  QRectF _rect_chart;
  const VortexSequenceMap *_vmap;

private:
  std::vector<std::set<int> > _slots;
  std::map<int, int> _slotmap;
};

#endif
