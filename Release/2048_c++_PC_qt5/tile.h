#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include<QPainter>
#include <QFont>
#include <QMessageBox>
#include<QMenuBar>
#include<QUrl>
#include <QDesktopServices>
/*declare an enum-type array.
 * Make sure each color corresponding a certain value
Since 2048 equals to 2^11, there are 11 different color.*/
enum colorPalatte{PeachPuff,Gold,Orange,Chocolate,Tomato,Red,HotPink,Violet,Cyan,DodgerBlue,Chartreuse};
//This class is intent to define the color of each tile and the rules of variation
class Tile : public QWidget{    //Inheriance from QWidget
    Q_OBJECT    //QObject Macro Declartion
 public:
    Tile(QWidget *parent = nullptr) : QWidget(parent){
        setFixedSize(100,100);
    }
    int getNumber(){  //Get the number via a function to avoid stake of variable directly access
        return number;
    }
    void setNumber(int n){
        number = n;
    }
protected:
    int number;     //The exact value of each tile.
    void paintEvent(QPaintEvent*);  //Color controlling function
};
#endif // WIDGET_H
