#include "game.h"
#include "tile.h"

core::core(QWidget *parent) : QWidget(parent)
{
    start();    //run start() to build the all initial conditions
    myMenu();   //run myMenu()
}
void core::start(){              //Program initiation
   for(int i = 0; i < column; ++i)
   {
       QVector<Tile*> rowlist; //Initial a 1D to tail after each row
       for(int j = 0; j < row; ++j){
           Tile *t = new Tile(this);   //a space of point-to-Tile
           t->setGeometry(100 + 100 * i, 100 + 100 * j, 100, 100); //Locate each tile on the window
           t->setNumber(0);    //Initialize the value
           rowlist.push_back(t);//add an element after the vector
       }
       _tileList.push_back(rowlist);   //add each rowlist on the each block of tilelist
   }
   init_Score();
   generation();//random spawn 2 tiles
   generation();
}

void core::restart(){

            for(int i = 0; i < column; ++i)
            {
                for(int j = 0; j < row; ++j){
                    _tileList[i][j]->setNumber(0); //reset all tile's number
                }
            }
            generation();//random respawn 2 tiles and score
            generation();
            init_Score();
}

void core::generation(){     //spawn a tile in a blank block
       srand(time(0)); //Plant a random seed
       int new_column, new_row;    //the position of new spawn
       new_column = rand()%column; //randomize the column
       new_row = rand()%row;       //randomize the row
       detectLose();   //all numbers get stucked, lose the game, terminate the function
       for(;_tileList[new_column][new_row]->getNumber() !=0;){ //end the loop until it finds a blank space
           new_column = rand()%column; //randomize the column and row
           new_row = rand()%row;
       }_tileList[new_column][new_row]->setNumber(b[rand()%2]); //Reaching the proper position and set the number randomizing among 2 and 4
   }
/********Failure and Victory response function*****************/
    void core::lose(){
    QMessageBox msgBox; //Instantiate a Messagebox-type object
    msgBox.setText("YOU LOST THE GAME!!! Try again?"); //The text shows up after the player loses the game
    msgBox.exec();//Messagebox shows up until the player clicks on "OK" button
    }
    void core::win(){
        QMessageBox winBox; //Instantiate a Messagebox-type object
        winBox.setText("Winner winner, chicken dinner!"); //The text shows up after the player wins the game
        winBox.exec();//Messagebox shows up until the player clicks on "OK" button
    }

    void core::detectLose(){  //Determine the game failed
        if(full()){     //judge whether all tiles are occupied with non-zero number
            if(horizonMovable()||verticalMovable()){    //determine whether it could be moved horizonly or vertically
                return ;    //All directions are movable. Game goes on!
            }else lose();   //No hope~ Lose the Game
        }
    }
/*********Detect whether all tiles are occupied by any nonzero number*********/
        bool core::full(){
            for(int i = 0; i < row; i++){
                for(int j = 0; j < column; j++){
                    if(0 == _tileList[j][i]->getNumber()){//The number is zero in row i, column j.
                        //qDebug()<<"Not full enough";
                        return false;//Still, there is a blank space at least
                    }if(2048 == _tileList[j][i]->getNumber()){//The player reaches the goal! 2048
                        win(); //Sign of game victory
                    }
                }
            }//qDebug() << "full";
            return true;  //No blank space left
        }

 /******Detect whether the tiles are movable on the horizon or vertical direction*******/
 bool core::horizonMovable(){
                for(int i = 0; i < row; i++){
                    for(int j = 0; j < column-1; j++){
                        if(_tileList[j][i]->getNumber() == _tileList[j+1][i]->getNumber()){
                           //Any pair of left and right adjacent numbers are equal
        //                    qDebug()<<"HorizonlyMovable: yes";
                            return true;//Movable
                        }
                    }
                }
        //        qDebug()<<"HorizonlyMovable: no";
                return false;//stucked
            }
bool core::verticalMovable(){
                for(int j = 0; j < column; j++){
                    for(int i = 0; i< row -1; i++){
                        if(_tileList[j][i]->getNumber() == _tileList[j][i+1]->getNumber()){
                            //Any pair of up and down adjacent numbers are equal
        //                    qDebug()<<"VM : yes";
                            return true;//Movable
                        }
                    }
                }
        //        qDebug()<<"VM : no";
                return false;//stucked
            }

int core::getScore(){            //Function access score to avoid variable directly access
   return score;
}
/*********Initial score after restart or first start-up****************/
void core::init_Score(){
    score = 0;
}

void core::myMenu(){//Game menu
    QMenuBar *menuBar = new QMenuBar(this); //Instantiate a QMenuBar object
    QMenu *file = new QMenu; // apply a QMenu pointer
    file =  menuBar->addMenu(tr("&File")); // add the "file" menu to the "menuBar"
    file->setTitle("Game"); //Set text title of menu
    QAction * act1 = new QAction("New Game(R)",file); //Set the first QAction
    file->addAction(act1);                            //Link the action to the menu
    QAction * act3 = new QAction("How To Play",file); //Set the 2nd QAction
    file->addAction(act3);                            //Link the action to the menu
    QAction * act4 = new QAction("About 2048",file);//Set the 3rd QAction
    file->addAction(act4);                            //Link the action to the menu
    QAction * act5 = new QAction("Exit(Esc)",file);//Set the 4th QAction
    file->addAction(act5);                            //Link the action to the menu
    connect(menuBar, SIGNAL(triggered(QAction*)), this, SLOT(updateItem(QAction*)));/*connect a signal with a slot
    When user click some action, it will trigger some event */
}

void core::paintEvent(QPaintEvent *)//paint the dynamic score data
{
    QPainter score(this);   // instantiate a QPainter object which points to this(Widget-class)
    score.setPen(QColor(132,172,230)); // configure the color of the font
    score.setBrush(QColor(20,30,40));//configure the color of the background
     score.drawRect(0,0,900,600); // draw a fullscreen rectangle as a background
    QFont font;   // instantiate a QFont object
    font.setFamily("Snap ITC"); // set the type of the font
    font.setPixelSize(50); //set the size of the font
    score.setFont(font); //configure the font
    score.drawText(200,80, QString("Score: %1").arg(this->score));// configure the coordinates and the contents of the fonts
    QPainter tips(this); //show the tips to the user
    QFont font2; // instantiate another QFont object
    font.setFamily("Kunstler Script"); //set a new type of the font
    font2.setPixelSize(20); //set the size of the font
    tips.setPen(QColor(230,180,230));//configure the color of the text pen
    tips.setFont(font2); //configure the font
    tips.drawText(5,595,QString("If you got stucked, press R to restart or click the GAME menu")); // configure the coordinates and the contents of the tips

}
void core::keyPressEvent(QKeyEvent *event){ //key press reaction function

    if(event->type() == QEvent::KeyPress){ // when pressed any of the key
        QKeyEvent *keyEvent = static_cast<QKeyEvent*>(event);   //instantiate a QKeyEvent-type object and converts it to a static type
        if((keyEvent->key() == Qt::Key_Left)||( keyEvent->key() == Qt::Key_A)){ //When pressed "←" or "A"
            left();  //leftward
            detectLose(); //detect user's failure
            update(); //Threads call redraw event
        }
        if((keyEvent->key() == Qt::Key_Right)|| (keyEvent->key() == Qt::Key_D)){//When pressed "→" or "D"
            right(); //rightward
            detectLose();//detect user's failure
            update(); //Threads call redraw event
        }
        if((keyEvent->key() == Qt::Key_Up)|| (keyEvent->key() == Qt::Key_W)){//When pressed "↑" or "W"
            up(); //upward
            detectLose();//detect user's failure
            update(); //Threads call redraw event
        }
        if((keyEvent->key() == Qt::Key_Down)|| (keyEvent->key() == Qt::Key_S)){//When pressed "↓" or "S"
            down(); //downward
            detectLose();//detect user's failure
            update(); //Threads call redraw event
        }
//        if(keyEvent->key() == Qt::Key_Space){ //Using for Debug, uncomment it when debug
//            update();
//            generation();
//        }
        if(keyEvent->key() == Qt::Key_R){ //When pressed "R"
            restart(); //restart the game
            update(); //Threads call redraw event
        }
        if(keyEvent->key() == Qt::Key_Escape){ //When pressed "Esc"
            QWidget::close(); //Shut down the program
        }
    }
}


