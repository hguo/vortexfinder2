#include <QApplication>
#include <QFileDialog>
#include "widget.h"
#include "mainWindow.h"

int main(int argc, char **argv)
{
  std::string filename_vortex, filename_trace; 
  
  if (argc == 1) {
    fprintf(stderr, "Usage: %s <vortex_file> [trace_file]\n", argv[0]); 
    return EXIT_FAILURE;
  } 
  if (argc >= 2) filename_vortex = argv[1];
  if (argc >= 3) filename_trace = argv[2]; 

  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

  CGLWidget *widget = new CGLWidget;
  widget->show(); 
  widget->LoadVortexObjects(filename_vortex); 
  widget->LoadFieldLines(filename_trace);

  return app.exec(); 
}


#if 0 // legacy
  QString filename = QFileDialog::getOpenFileName(NULL, "Open vortex file...", "./", "*.vortex");
  filename = filename1.toStdString(); 
#endif
