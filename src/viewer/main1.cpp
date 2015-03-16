#include <QApplication>
#include <QFileDialog>
// #include <GLUT/glut.h>
#include "widget.h"
#include "mainWindow.h"

int main(int argc, char **argv)
{
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <dataname> <ts> <tl>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string dataname = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]);

  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

  VortexTransition vt;
  VortexSequenceMap vmap;
  vt.LoadFromFile(dataname, ts, tl);
  vmap.Construct(vt, ts, tl);

  CGLWidget *widget = new CGLWidget;
  widget->show();
  widget->SetData(dataname, ts, tl);
  widget->SetSequenceMap(&vmap);
  widget->LoadTimeStep(ts);

  return app.exec(); 
}

