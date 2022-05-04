/*
 * @Author: FrankTudor
 * @Date: 2022-05-04 00:51:30
 * @LastEditors: FrankTudor
 * @Description: This file is created, edited, contributed by FrankTudor
 * @LastEditTime: 2022-05-04 23:16:45
 */
#include<SFML/Graphics.hpp>
#include<time.h>
#include<iostream>
#include<string>
#include<filesystem>
#define WIDTH 400
#define HEIGHT 533
#define DEATH_HEIGHT 508
#define VINCIBLE
using namespace sf;
using std::filesystem::current_path;
struct point{
	int x,y;
};
point plat[20];
int x=100,y=100,h=200;
float dx=0,dy=0;
int jumpHeight = 0;
int lastPlatformId = -1;
int lastPlatformHeight = 0;
int platformCount = 0;

void init(){
	h=200;
    dx=0,dy=0;
    for (int i=0;i<10;i++){
       plat[i].x=rand()%400;
       plat[i].y=rand()%533;
	}
	x = plat[0].x+20;
	y = plat[0].y-20;
	jumpHeight = 0;
	lastPlatformId = -1;
	lastPlatformHeight = DEATH_HEIGHT ;
	platformCount = 0;
}

int main(){
	std::string dir = current_path();
	srand(time(NULL));
	RenderWindow app(VideoMode(WIDTH,HEIGHT), "Doodle Jump!");
	app.setFramerateLimit(60);
	 
	Texture tBackground, tDoodle, tPlatform, tGameOver;
	tBackground.loadFromFile(dir + "/../images/background.png");
	tDoodle.loadFromFile(dir + "/../images/doodle-jumper.png");
	tPlatform.loadFromFile(dir + "/../images/platform.png");
	tGameOver.loadFromFile(dir + "/../images/gameOver.png");

	Sprite  sDoodle(tDoodle), sPlatform(tPlatform), sGameOver(tGameOver);
	Sprite sBackground(tBackground);
	init();
	sGameOver.setPosition(100,100);



	Font font;
	if(!font.loadFromFile(dir + "/../Font/Roboto-Black.ttf")){
		throw  std::runtime_error("Font loading failed");
	}
	String showTex = "Score: 0";
	Text score(showTex,font);
	score.setCharacterSize(30);
	score.setFillColor(Color::Blue);
	score.setStyle(Text::Bold);
	score.setPosition(0,0);

    while (app.isOpen()){
		app.clear();
		app.draw(sBackground);

        Event e;
        while (app.pollEvent(e)){
            if (e.type == Event::Closed)
                app.close();
        }

		if (Keyboard::isKeyPressed(Keyboard::Right)){x+=3;}//right
		if (Keyboard::isKeyPressed(Keyboard::Left)){x-=3;}//left

		dy+=0.2;
		y+=dy;
		if (y>500)  dy=-10; //downward

		if (y<h){
			for (int i=0;i<10;i++){
				y=h;
				plat[i].y=plat[i].y-dy;
				if (plat[i].y>533) {plat[i].y=0; plat[i].x=rand()%400;}
			}
		}
		int curPlatformHeight = 0;
		for (int i=0;i<10;i++){// jump on the platform
			if ((x+50>plat[i].x) && (x+20<plat[i].x+68)
			&& (y+70>plat[i].y) && (y+70<plat[i].y+14) && (dy>0)){
				curPlatformHeight = (plat[i].y);
				// printf("%d: platform height: %d || %d: last height: %d\n", i ,curPlatformHeight, lastPlatformId, lastPlatformHeight);
				if(lastPlatformId != i){
					if(jumpHeight < 0){jumpHeight = 0;}//it's a bug, sometimes height is below zero
					lastPlatformId = i;
					jumpHeight+= -(curPlatformHeight - lastPlatformHeight);
					lastPlatformHeight += curPlatformHeight;
					++platformCount;
					// printf("score: %d, %d jumps\n", jumpHeight/10, platformCount);
					
				}
				dy=-10;
			}  
		}
		if(x < -80){x = x + 460;}//x scope from -80 to 380
		if(x > 380){x = x - 460;}

		sDoodle.setPosition(x,y);

		app.draw(sDoodle);
		for (int i=0;i<10;i++){
			sPlatform.setPosition(plat[i].x,plat[i].y);
			app.draw(sPlatform);
		}
#ifdef VINCIBLE
	if(y >= DEATH_HEIGHT){//Death handler
		app.clear();
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
		String a = "a %d";
		
		showTex = std::string("Score: ") + std::to_string(jumpHeight/10) + std::string(", ") + std::to_string(platformCount) + std::string(" Jumps");
		score.setString(showTex);
		app.draw(score);
		app.display();
	}

	return 0;
 }