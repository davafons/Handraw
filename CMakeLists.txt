cmake_minimum_required(VERSION 2.8)
project( Handraw )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( Handraw main.cpp MyBGSubtractorColor.cpp )
target_link_libraries( Handraw ${OpenCV_LIBS} )
