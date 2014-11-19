#ifndef _WIDGET_H
#define _WIDGET_H

#include <QGLWidget>
#include <QList>
#include <cmath>
#include "trackball.h"

class QMConnector; 
class QMouseEvent;
class QKeyEvent; 
class QWheelEvent; 

class CGLWidget : public QGLWidget
{
  Q_OBJECT

public:
  CGLWidget(const QGLFormat& fmt=QGLFormat::defaultFormat(), QWidget *parent=NULL, QGLWidget *sharedWidget=NULL); 
  ~CGLWidget(); 

protected:
  void initializeGL(); 
  void resizeGL(int w, int h); 
  void paintGL(); 

  void mousePressEvent(QMouseEvent*); 
  void mouseMoveEvent(QMouseEvent*);
  void keyPressEvent(QKeyEvent*); 
  void wheelEvent(QWheelEvent*); 

private:
  CGLTrackball _trackball;
  
private: // camera
  const float _fovy, _znear, _zfar; 
  const QVector3D _eye, _center, _up;

  std::vector<float> _points; 
}; 

#endif
