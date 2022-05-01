#include "tile.h"
#include"game.h"
#include <QWidget>
#include <QKeyEvent>
#include <math.h>
void Tile::paintEvent(QPaintEvent *){ //paint each tile of the 4-by-4 grid
    QPainter painter(this); //Instantiate a QPainter object
    switch (int(log(double(number))/log(2))) { //Convert number into the exponent of the number
    case PeachPuff: painter.setBrush(QColor::fromRgb(255,218,185));break;  //Different exponents correspond to different colors one by one
    case Gold: painter.setBrush(QColor::fromRgb(255,210,0));break;
    case Orange: painter.setBrush(QColor::fromRgb(255,165,0));break;
    case Chocolate: painter.setBrush(QColor::fromRgb(210,105,30));break;
    case Tomato: painter.setBrush(QColor::fromRgb(255,99,71));break;
    case Red: painter.setBrush(QColor::fromRgb(255,0,0));break;
    case HotPink: painter.setBrush(QColor::fromRgb(255,105,180));break;
    case Violet: painter.setBrush(QColor::fromRgb(238,130,238));break;
    case Cyan: painter.setBrush(QColor::fromRgb(0,255,255));break;
    case DodgerBlue: painter.setBrush(QColor::fromRgb(30,144,255));break;
    case Chartreuse: painter.setBrush(QColor::fromRgb(102,205,0));break;
    default: painter.setBrush(QColor::fromRgb(250,235,215)); //AntiqueWhite
    }
    painter.setPen(Qt::NoPen);//Do not draw the border of the color block
    painter.drawRoundRect(5,5,94,94,50,50);//The relative coordinates and shape and the size of the colored tile block
    QPen textPen;//Instantiate a QPen object
    textPen.setWidth(100);
    textPen.setColor(Qt::black);// color
    painter.setPen(textPen); //configure the painter's pen
    if(0 != number){ //show the number of the nonzero value
    QFont font; //instantiate a QFont object
    font.setFamily("BERNIER Regular"); //set the type of the font
    font.setPointSize(20);      //set the size of the number text
    painter.setFont(font);      //configrue the font
    painter.drawText(20,50, QString("%1").arg(number)); //configure the coordinates and the contents of the text
    }
   if(2048 == number){ //when someone win the game
       QMessageBox msgBox; //instantiate a QMessageBox-type object
       msgBox.setText("You Won!"); //the text contents it shows
       msgBox.exec();
   }
}
