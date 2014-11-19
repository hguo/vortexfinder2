#include <QApplication>
#include <glut.h>
#include "widget.h"

int main(int argc, char **argv)
{
  glutInit(&argc, argv); 
  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

  CGLWidget *widget = new CGLWidget;
  widget->show(); 

  return app.exec(); 
}
