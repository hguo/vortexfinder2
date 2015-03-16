#include <QApplication>
#include <QFileDialog>
// #include <GLUT/glut.h>
#include "widget.h"
#include "mainWindow.h"

int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <filename> <timestep>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string filename = argv[1];
  int timestep = atoi(argv[2]);

  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

  CGLWidget *widget = new CGLWidget;
  widget->show();
  widget->SetDataName(filename);
  // widget->OpenGLGPUDataset();
  widget->LoadTimeStep(timestep);

  return app.exec(); 
}

