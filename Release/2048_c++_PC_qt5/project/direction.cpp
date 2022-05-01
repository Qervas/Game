#include"game.h"
/*******Direction Keyboard Pressed Reaction Function Specification************/
/*
* Set 2 rows of four numbers: {2, 0, 4, 4}
*                             {0, 2, 2, 8}
*Operating 2 watchposts, say t1 and t2, keeping them in certain order:
*  in left(), t1 < t2; right(); t1 > t2; up(), t1 < t2; down(), t1 > t2.
*
*Observer variable: flag #If the number in the tiles has been moved, set flag = 1
*                        #Otherwise set flag = 0
*                        #When flag = 1, at the end of the function, it will randomly generate 2 or 4 in two random tiles
*
*Detector: In each row, when t2 reaches the boundary and the
*          operation is completed, enter the operation of the next row.
*          End the function after all 4 lines are traversed.
*
*Steps: #Take the left() as the example
*  1.Initialization: t1 = x1, t2 = x2, flag = 0
*   { 2,  0,   4,  4}
*    ||   ||
*    t1   t2
*   { 0,  2,   2,  8}
*    ||   ||
*    t1   t2
*  2.Judgment and Order Rearrangement:
*   1) if t2 == 0 or t1 == t2(always after t1++ operation),then t2 moves to the next position legally.
*     {2,  0,  4,  4}
*      ||      ||
*      t1      t2
*     Go to step2.2)
*
*   2)Detect first nonzero number at t2, then check if t1 is zero.
*     a)When t1 is zero(ensured T2 is non-zero), assign t2's value to t1's and then assign 0 to t2.
*        { 0,  2,  2,  8}  ----->  { 2,  0,  2,  8}
*         ||  ||                     ||  ||
*         t1  t2                     t1  t2
*     Go back to step2.1)
*
*     b)When t1 is non-zero(ensured T2 is non-zero too), go to step3.
*
*   3.Comparison and Combination: Compare the numbers at t1 and t2 for equality
*     1) [t1] != [t2], 2 designated number are to be placed adjacently(put t2-number at [t1+1])
*       a) When t1 + 1 == t2,(it's venturous, dangerous to operate the number value directly
*          without this conditional judgment) NO ADDITIONAL OPERATION REQUIERD!!!
*          t1 moves to the next position legally. Go to step2.1)
*          { 2, 4, 8, 16}   -----> { 2, 4, 8, 16}
*            || ||                      ||
*            t1 t2                     t1,t2
*
*       b) When t1 + 1 != t2, there is a distance between t1 and t2.
*         - Assign t2's value to [t1+1]'s
*         - Set t2's as 0
*         - Set flag = 1
*         - t1 moves to the next position legally. Go to step2.1)
*           { 2, 0, 4, 4}  ----> { 2, 4, 0, 4} ---->{ 2, 4, 0, 4}
*             ||    ||             ||    ||              || ||
*             t1    t2             t1    t2              t1 t2
*
*     2) [t1] == [t2],
*       - Double the value of t1,
*       - Set t2's value as 0
*       - The total score plus doubled t1
*       - Set flag = 1
*       - Go to step2.1)
*           { 2, 4, 0, 4} ----> { 2, 8, 0, 0}  { 2, 0, 2, 8} ----> { 4, 0, 0, 8}
*                ||   ||             ||   ||     ||    ||            ||    ||
*                t1   t2             t1   t2     t1    t2            t1    t2
*
*/


void core::up(){  //When pressed up keyboard, how the program should be react.
//    qDebug()<<"\npressed up";
        int t1,t2;  //2 watchposts
        int flag = 0;  //Observer of tiles movement
    for(int i = 0; i < row; i++){ //restrict the operation within determined scope
        t1 = 0, t2 = 1; //initialize the watchposts
        for (;t2<column;){ //
            if((0 == _tileList[i][t2]->getNumber()) ||t1 == t2){ //Judgement and Rearrangement
                t2++; //seeking the first non-zero number
            }else if(_tileList[i][t2]->getNumber()){ //Captured the first non-zero number at t2 location
                if(0 == _tileList[i][t1]->getNumber()){ // number on t1 is zero
                    _tileList[i][t1]->setNumber(_tileList[i][t2]->getNumber()); //  t2's value replaces t1's
                    _tileList[i][t2]->setNumber(0); // reset value of t2
                    flag = 1; //movement detected
                }else if(_tileList[i][t1]->getNumber()){ // number on t1 is nonzero
                    if(_tileList[i][t1]->getNumber() != _tileList[i][t2]->getNumber()){// [t1] != [t2]
                        if(t1+1 == t2){ //t2 has already next to t1, then do nothing
                        }else{_tileList[i][t1+1]->setNumber(_tileList[i][t2]->getNumber()); //assign t2's value to the [t1+1]'s
                        _tileList[i][t2]->setNumber(0); //reset value of t2
                        flag = 1; //movement detected
                        }
                        t1++; //nudge t1 a tile to the right
                    }else if(_tileList[i][t1]->getNumber() == _tileList[i][t2]->getNumber()){ // [t1] == [t2]
                        _tileList[i][t1]->setNumber(_tileList[i][t1]->getNumber() << 1); //double the value of t1
                        score += _tileList[i][t1]->getNumber(); // sum score
                        _tileList[i][t2]->setNumber(0); //reset the t2's value
                        t1++; //nudge t1 a tile to the next position
                        flag = 1; //movement detected
                    }
                }
            }
            }
        }
    if(flag){ //movement detector triggered
    generation(); //randomly generate a number at a random position
    }
}
void core::down(){
//    qDebug()<<"\npressed down";
        int t1,t2;//2 watchposts
        int flag = 0;//Observer of tiles movement
    for(int i = 0; i < row; i++){//restrict the operation within determined scope
        t1 = row -1, t2 = row -2;//initialize the watchposts
        for (;t2>=0;){
            if((0 == _tileList[i][t2]->getNumber()) ||t1 == t2){ //Judgement and Rearrangement
                t2--; //seeking the first non-zero number
            }else if(_tileList[i][t2]->getNumber()){ //Captured the first non-zero number at t2 location
                if(0 == _tileList[i][t1]->getNumber()){ // number on t1 is zero
                    _tileList[i][t1]->setNumber(_tileList[i][t2]->getNumber()); //  t2's value replaces t1's
                    _tileList[i][t2]->setNumber(0); // reset t2's value
                    flag = 1; //movement detected
                }else if(_tileList[i][t1]->getNumber()){ // number on t1 is nonzero
                    if(_tileList[i][t1]->getNumber() != _tileList[i][t2]->getNumber()){// [t1] != [t2]
                        if(t1-1 == t2){ //t2 has already next to t1, then do nothing
                        }else{_tileList[i][t1-1]->setNumber(_tileList[i][t2]->getNumber()); //assign t2's value to the [t1+1]'s
                        _tileList[i][t2]->setNumber(0); //reset value of t2
                        flag = 1;  //movement detected
                        }
                        t1--; //nudge t1 a tile to the right
                    }else if(_tileList[i][t1]->getNumber() == _tileList[i][t2]->getNumber()){  // [t1] == [t2]
                        _tileList[i][t1]->setNumber(_tileList[i][t1]->getNumber() <<1); //double the value of t1
                        score += _tileList[i][t1]->getNumber(); // sum score
                        _tileList[i][t2]->setNumber(0); //reset the t2's value
                        t1--; //nudge t1 a tile to the next position
                        flag = 1; //movement detected
                    }
                }
            }
            }
        }
    if(flag){//movement detector triggered
        generation();//randomly generate a number at a random position
    }
}
void core::right(){
//    qDebug()<<"\npressed right";
        int t1,t2;  //2 watchposts
        int flag = 0;    //Observer of tiles movement
    for(int i = 0; i < column; i++){    //restrict the operation within determined scope
        t1 = column -1, t2 = column -2; //initialize the watchposts
        for (;t2>=0;){
            if((0 == _tileList[t2][i]->getNumber()) ||t1 == t2){ //Judgement and Rearrangement
                t2--;  //seeking the first non-zero number
            }else if(_tileList[t2][i]->getNumber()){ //Captured the first non-zero number at t2 location
                //set t1 and t2
                if(0 == _tileList[t1][i]->getNumber()){ // m[t1] is empty
                    _tileList[t1][i]->setNumber(_tileList[t2][i]->getNumber());//  t2's value replaces t1's
                    _tileList[t2][i]->setNumber(0);// reset t2's value
                    flag = 1; //movement detected
                }else if(_tileList[t1][i]->getNumber()){  // number on t1 is nonzero
                    if(_tileList[t1][i]->getNumber() != _tileList[t2][i]->getNumber()){// [t1] != [t2]
                        if(t1-1 == t2){ //t2 has already next to t1, then do nothing
                        }else{_tileList[t1-1][i]->setNumber(_tileList[t2][i]->getNumber()); //assign t2's value to the [t1+1]'s
                        _tileList[t2][i]->setNumber(0); //reset value of t2
                        flag = 1; //movement detected
                        }
                        t1--; //nudge t1 a tile to the right
                    }else if(_tileList[t1][i]->getNumber() == _tileList[t2][i]->getNumber()){// [t1] == [t2]
                        _tileList[t1][i]->setNumber(_tileList[t1][i]->getNumber() <<1);
                        score += _tileList[t1][i]->getNumber();// sum score
                        _tileList[t2][i]->setNumber(0); //nudge t1 a tile to the right
                        t1--; //nudge t1 a tile to the next position
                        flag = 1; //movement detected
                    }
                }
            }
            }
        }
    if(flag){//movement detector triggered
    generation();//randomly generate a number at a random position
    }
}
void core::left(){//to read exact instruction check the bottom of the file
//    qDebug()<<"\npressed left";
        int t1,t2; //subscripts for each item comparison
        int flag = 0; //Observer of tiles movement
    for(int i = 0; i < column; i++){//restrict the operation within determined scope
        t1 = 0, t2 = 1;//initialize the watchposts
        for (;t2<column;){
            if((0 == _tileList[t2][i]->getNumber()) ||t1 == t2){ //Judgement and Rearrangement
                t2++; //seeking the first non-zero number
            }else if(_tileList[t2][i]->getNumber()){//Captured the first non-zero number at t2 location
                if(0 == _tileList[t1][i]->getNumber()){ // number on t1 is zero
                    _tileList[t1][i]->setNumber(_tileList[t2][i]->getNumber()); // t2's value replaces t1's
                    _tileList[t2][i]->setNumber(0); // reset t2's value
                    flag = 1; //movement detected
                }else if(_tileList[t1][i]->getNumber()){ // number on t1 is nonzero
                    if(_tileList[t1][i]->getNumber() != _tileList[t2][i]->getNumber()){//[t1] != [t2]
                        if(t1+1 == t2){ //t2 has already next to t1, then do nothing
                        }else{_tileList[t1+1][i]->setNumber(_tileList[t2][i]->getNumber()); //assign t2's value to the [t1+1]'s
                        _tileList[t2][i]->setNumber(0); //reset value of t2
                        flag = 1;//reset value of t2
                        }
                        t1++;//nudge t1 a tile to the right
                    }else if(_tileList[t1][i]->getNumber() == _tileList[t2][i]->getNumber()){// [t1] == [t2]
                        _tileList[t1][i]->setNumber(_tileList[t1][i]->getNumber() <<1); //double the value of t1
                        score += _tileList[t1][i]->getNumber();// sum score
                        _tileList[t2][i]->setNumber(0);//reset the t2's value
                        t1++;//nudge t1 a tile to the right
                        flag = 1;//movement detected
                    }
                }
            }
            }
        }
    if(flag){//movement detector triggered
    generation();//randomly generate a number at a random position
    }
}
