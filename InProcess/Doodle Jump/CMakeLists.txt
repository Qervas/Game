
project( DooleJump )


cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
add_library(dooleJump main.cpp)
add_executable( ${PROJECT_NAME} main.cpp )
target_link_libraries( ${PROJECT_NAME} -lsfml-graphics -lsfml-window -lsfml-system )
