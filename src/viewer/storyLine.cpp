#include <QApplication>
#include "storyLineWidget.h"

int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <dataname> <ts> <tl>\n", argv[0]);
    return EXIT_FAILURE;
  }
  
  const std::string dataname = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]), 
            span = 1;

  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 
  
  VortexTransition vt;
  VortexSequenceMap vmap;
  vt.LoadFromFile(dataname, ts, tl);
  vmap.Construct(vt, ts, tl);

  CStorylineWidget *widget = new CStorylineWidget;
  widget->SetSequenceMap(&vmap);
  widget->show();

  return app.exec(); 
}
