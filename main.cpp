#include <string>

#include <opencv2/opencv.hpp>

#include "MyBGSubtractorColor.h"

// Function declaration
void open_camera(cv::VideoCapture &cap);
void handle_input(int c);


// Global variables
const int MAX_EMPTY_FRAMES_TO_READ = 2000;

bool quit = false;




int main() {
  cv::VideoCapture cap;

  const std::string reconocimiento = "Reconocimiento";
  cv::namedWindow(reconocimiento);

  try {
    open_camera(cap);

    MyBGSubtractorColor bg_sub;
    bg_sub.LearnModel(cap);

    // MAIN LOOP
    while (!quit) {
      cv::Mat frame;
      cap >> frame;
      cv::flip(frame, frame, 1);

      imshow(reconocimiento, frame);

      handle_input(cv::waitKey(40));
    }

  } catch (const std::runtime_error &e) {
    std::cerr << "ERROR::" << e.what() << std::endl;
  }

  cv::destroyWindow(reconocimiento);
  cap.release();

  return 0;
}


void open_camera(cv::VideoCapture &cap) {
  if (!cap.open(0))
    throw std::runtime_error("No se pudo abrir la cámara!");

  cv::Mat frame;

  int cont = 0;
  while (frame.empty() && ++cont < 2000)
    cap >> frame;

  if(cont >= MAX_EMPTY_FRAMES_TO_READ)
    throw std::runtime_error("No se ha podido leer un frame valido tras iniciar la cámara!");
}


void handle_input(int c) {
  switch(c) {
    case 27: // Escape key
    case 'q':
      quit = true;
      break;


  }
}

