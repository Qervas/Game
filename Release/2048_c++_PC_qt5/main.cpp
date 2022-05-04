#include "tile.h"
#include "game.h"
#include <QApplication>

int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
    core w;   //Instantiate a window object
    w.setFixedSize(600,600);    //set a fixed size of window
    w.show();   //show the window
    return a.exec();
}
