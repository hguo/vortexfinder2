#include <QApplication>
#include <QFileDialog>
#include "widget.h"
#include "mainWindow.h"

int main(int argc, char **argv)
{
  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

#if 0
  CMainWindow *mainWindow = new CMainWindow;
  mainWindow->show();
#else
  std::string filename;
  if (argc>1) 
    filename = argv[1];
  else {
    QString filename1 = QFileDialog::getOpenFileName(NULL, "Open vortex file...", "./", "*.vortex");
    filename = filename1.toStdString(); 
  }

  CGLWidget *widget = new CGLWidget;
  widget->show(); 
  widget->LoadVortexObjects(filename); 
#endif

  return app.exec(); 
}
