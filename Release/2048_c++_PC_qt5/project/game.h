#ifndef CORE_H
#define CORE_H

#include <QObject>
#include "tile.h"
#include<QPainter>
#include <QMessageBox>
#include<QAction>
#include<QMenuBar>
#include<QUrl>
#include <QDesktopServices>
#include<QKeyEvent>
class core : public QWidget
{
    Q_OBJECT
public:
    core(QWidget *parent = nullptr);
    void start();
    void restart();
    void generation();
    void lose();
    void win();
    void detectLose();
    bool full();
    bool verticalMovable();
    bool horizonMovable();
    void up();
    void down();
    void left();
    void right();
    void myMenu();
private:
    int column = 4, row = 4;
protected:
    int i,j;   // variable used as for() circulation counter
    int b2=0b000000000010;     //equals to 2(10 base)
    int b4 = b2 << 1;          //equals to 4(10 base)
    int b[2]{b2,b4};           //random value initiation array
    int score=0;
    QVector<QVector<Tile *>> _tileList; //Build a 2D vector
    int getScore();
    void init_Score();
    void keyPressEvent(QKeyEvent *event) ; //Override keyPressEvent() function
    void paintEvent(QPaintEvent *event) ;  // Override painEvent() function
signals:
    void triggered(QAction*);   //Declare a signal for the menu
public slots:
    void updateItem(QAction* action){
        /*Menu Reaction Function
        When user click the contents which are quoted("") in the condition statements,
        this is how the program reacts.*/
        if(action->text() == "About 2048"){ //When clicked "About 2048"
            QUrl Url("https://2048game.com/tips-and-tricks/");//Instantiate a url-object and put the link into it
            QDesktopServices::openUrl(Url);//Lead the user to the web
        }
        if(action->text() == "How To Play"){ //When clicked "How to Play"
            QMessageBox msgBox;//Instantiate a Messagebox-type object
            //Show the text about how to play the game
            msgBox.setText("Use your arrow keys \" ↑↓←→ \" to move the tiles. When two tiles with the same number touch, they merge into one!");
            msgBox.exec();//Execute
        }
        if(action->text() == "New Game(R)"){ //When clicked "New Game(R)"
            restart();//restrat the game
            update();
        }
        if(action->text() == "Exit(Esc)"){ //When clicked "EXit(Esc)"
            QWidget::close();//shutdown the program
        }
    }
};

#endif // CORE_H
