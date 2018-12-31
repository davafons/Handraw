# Project
cmake_minimum_required(VERSION 3.6)
project( Handraw )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 14)

# Sources
file(GLOB SOURCES "src/*.cpp")

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Binary
add_executable( Handraw ${SOURCES} )
target_link_libraries( Handraw ${OpenCV_LIBS} )

