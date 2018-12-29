#include <string>

#include <opencv2/opencv.hpp>

#include "MyBGSubtractorColor.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

// Function declaration
void open_camera(cv::VideoCapture &cap);
void correct_median_size(int, void *userdata);
void handle_input(int c);

// Global variables
const int MAX_EMPTY_FRAMES_TO_READ = 2000;

bool quit = false;

cv::VideoCapture cap;
MyBGSubtractorColor bg_sub;

int main() {

  // 1º - Open camera
  try {
    open_camera(cap);
  } catch (const std::runtime_error &e) {
    std::cerr << "ERROR::" << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // 2º - Learn Model using samples from camera
  bg_sub.LearnModel(cap);

  // 3º - Create windows
  const std::string reconocimiento = "Reconocimiento";
  const std::string fondo = "Fondo";
  cv::namedWindow(reconocimiento);
  cv::namedWindow(fondo);

  int dilation_size = 2;
  cv::createTrackbar("Dilation size:", fondo, &dilation_size, 40, nullptr);

  int median_size = 5;
  cv::createTrackbar("Median size:", fondo, &median_size, 40,
                     correct_median_size, &median_size);

  // MAIN LOOP
  while (!quit) {
    handle_input(cv::waitKey(40));

    cv::Mat frame;
    cap >> frame;

    // 4º - Background subtraction
    cv::Mat bgmask;
    bg_sub.ObtainBGMask(frame, bgmask);

    // 5º - Noise reduction
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {2 * dilation_size + 1, 2 * dilation_size + 1});

    cv::morphologyEx(bgmask, bgmask, cv::MORPH_OPEN, element);
    cv::medianBlur(bgmask, bgmask, median_size);
    /* cv::dilate(bgmask, bgmask, cv::Mat(), cv::Point(-1, -1), 3); */

    // Show windows
    cv::flip(frame, frame, 1);
    cv::imshow(reconocimiento, frame);
    cv::imshow(fondo, bgmask);
  }

  cv::destroyWindow(reconocimiento);
  cv::destroyWindow(fondo);
  cap.release();

  return EXIT_SUCCESS;
}

void open_camera(cv::VideoCapture &cap) {
  if (!cap.open(0))
    throw std::runtime_error("No se pudo abrir la cámara!");

  cv::Mat frame;
  int cont = 0;
  while (frame.empty() && ++cont < 2000)
    cap >> frame;

  if (cont >= MAX_EMPTY_FRAMES_TO_READ)
    throw std::runtime_error(
        "No se ha podido leer un frame valido tras iniciar la cámara!");
}

void correct_median_size(int, void *userdata) {
  int &median_size = *reinterpret_cast<int *>(userdata);
  if (median_size % 2 == 0)
    ++median_size;
}

void handle_input(int c) {
  switch (c) {
  case 27: // Escape key
  case 'q':
    quit = true;
    break;

  case 'r':
    bg_sub.LearnModel(cap);
    break;
  }
}
