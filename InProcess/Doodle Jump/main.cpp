/*
 * @Author: FrankTudor
 * @Date: 2022-05-04 00:51:30
 * @LastEditors: FrankTudor
 * @Description: This file is created, edited, contributed by FrankTudor
 * @LastEditTime: 2022-05-05 12:39:08
 */
#include<SFML/Graphics.hpp>
#include<time.h>
#include<iostream>
#include<string>
#include<filesystem>
#define BACKGROUND_WIDTH 400
#define BACKGROUND_HEIGHT 533
#define PLATFORM_WIDTH 68
#define PLATFORM_HEIGHT 14
#define DOODLE_WIDTH 74
#define DOODLE_HEIGHT 75
#define DEATH_HEIGHT 508
#define RIGHT_HAND_SIDE 1
#define LEFT_HAND_SIDE 0


#define VINCIBLE
using namespace sf;
using std::filesystem::current_path;
struct point{
	int x,y;
};
point plat[20];
int x=100,y=100,h=200;
float dx=0,dy=0;
int jumpScore = 0;
int lastPlatformId = -1;
int lastPlatformHeight = 0;
int platformCount = 0;
int highestScore = 0;
int highestJump = 0;
bool direction = RIGHT_HAND_SIDE;
int curPlatformHeight = 0;

void init(){
	h=200;
    dx=0,dy=0;
    for (int i=0;i<10;i++){
       plat[i].x=rand()%400;
       plat[i].y=rand()%533;
	}
	x = plat[0].x+20;
	y = plat[0].y-20;
	jumpScore = 0;
	lastPlatformId = -1;
	lastPlatformHeight = DEATH_HEIGHT ;
	platformCount = 0;
	curPlatformHeight = 0;

}

int main(){
	std::string dir = current_path();
	srand(time(NULL));
	RenderWindow app(VideoMode(BACKGROUND_WIDTH,BACKGROUND_HEIGHT), "Doodle Jump!");
	app.setFramerateLimit(60);
	 
	Texture tBackground, tDoodle, tPlatform, tGameOver;
	tBackground.loadFromFile(dir + "/../images/background.jpg");
	tDoodle.loadFromFile(dir + "/../images/doodle_jumper.png");
	tPlatform.loadFromFile(dir + "/../images/platform_backup.png");
	tGameOver.loadFromFile(dir + "/../images/gameOver.png");

	Sprite  sDoodle(tDoodle), sPlatform(tPlatform), sGameOver(tGameOver);
	Sprite sBackground(tBackground);
	init();
	sGameOver.setPosition(100,100);



	Font font;
	if(!font.loadFromFile(dir + "/../Font/Tapestry-Regular.ttf")){
		throw  std::runtime_error("Font loading failed");
	}
	String showTex;
	Text score("",font);
	score.setCharacterSize(30);
	score.setFillColor(Color::Magenta);
	score.setStyle(Text::Bold);
	score.setPosition(0,0);

	Text record("",font);
	record.setCharacterSize(40);
	record.setFillColor(Color::White);
	
    while (app.isOpen()){
		app.clear();
		app.draw(sBackground);

        Event e;
        while (app.pollEvent(e)){
            if (e.type == Event::Closed)
                app.close();
        }

		if (Keyboard::isKeyPressed(Keyboard::Right)){
			// if(direction != RIGHT_HAND_SIDE){
			// 	direction = RIGHT_HAND_SIDE;
			// 	sDoodle.setScale(-1,1);
			// }
			x+=3;

		}//right
		if (Keyboard::isKeyPressed(Keyboard::Left)){
			// if(direction != LEFT_HAND_SIDE){
			// 	direction = LEFT_HAND_SIDE;
			// 	sDoodle.setScale(-1,1);
			// }
			x-=3;
		}//left

		dy+=0.2;
		y+=dy;
		if (y>500)  dy=-10; //downward

		if (y<h){
			for (int i=0;i<10;i++){
				y=h;
				plat[i].y=plat[i].y-dy;
				if (plat[i].y>BACKGROUND_HEIGHT) {plat[i].y=0; plat[i].x=rand()%BACKGROUND_WIDTH;}//generate new platform
			}
		}
		
		for (int i=0;i<10;i++){// jump on the platform
			if ((x+50>plat[i].x) && (x+20<plat[i].x+PLATFORM_WIDTH)
			&& (y+70>plat[i].y) && (y+70<plat[i].y+PLATFORM_HEIGHT) && (dy>0)){
				curPlatformHeight = DEATH_HEIGHT - plat[i].y;
				// printf("%d: platform height: %d || %d: last height: %d\n", i ,curPlatformHeight, lastPlatformId, lastPlatformHeight);
				if(lastPlatformId != i){
					if(jumpScore < 0){jumpScore = 0;}//it's a bug, sometimes height is below zero
					lastPlatformId = i;
					jumpScore+= curPlatformHeight;
					++platformCount;
					// printf("score: %d, %d jumps\n", jumpScore/10, platformCount);
					
				}
				dy=-10;
			}  
		}
		if(x < -DOODLE_WIDTH){x = x + 460;}//x scope from -80 to 380
		if(x > BACKGROUND_WIDTH - 100 + DOODLE_WIDTH){x = x - 460;}

		sDoodle.setPosition(x,y);

		app.draw(sDoodle);
		for (int i=0;i<10;i++){
			sPlatform.setPosition(plat[i].x,plat[i].y);
			app.draw(sPlatform);
		}
#ifdef VINCIBLE
	if(y >= DEATH_HEIGHT){//Death handler
		app.clear();
		showTex = std::string("Record: ") + std::to_string(highestScore) + std::string("m\n\t\t\t  ") + std::to_string(highestJump) + std::string(" Jumps");

		record.setString(showTex);
		app.draw(record);
		app.draw(sGameOver);
		app.display();
		while(1){
			if(Keyboard::isKeyPressed(Keyboard::R)){break;}
			while (app.pollEvent(e)){
				if (e.type == Event::Closed)
					app.close();
			}
		}
		init();
		continue;
		
	}		
#endif
		highestScore = highestScore < jumpScore/120 ? highestScore + jumpScore/120 : highestScore;
		highestJump = highestJump < platformCount ? platformCount : highestJump;
		showTex = std::string("Score: ") + std::to_string(jumpScore/120) + std::string("m ,") + std::to_string(platformCount) + std::string(" Jumps");
		score.setString(showTex);
		app.draw(score);
		app.display();
	}

	return 0;
 }