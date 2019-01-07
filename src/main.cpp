#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "HandGesture.h"
#include "MyBGSubtractorColor.h"

// Function declaration
void open_camera(cv::VideoCapture &cap);
void correct_median_size(int, void *userdata);
void handle_input(int c);

// Global variables
const int MAX_EMPTY_FRAMES_TO_READ = 2000;

bool quit = false;

cv::VideoCapture cap;
MyBGSubtractorColor bg_sub;
HandGesture hand_detector;

int main(int argc, char *argv[]) {

  // 1º - Open camera
  try {
    open_camera(cap);
  } catch (const std::runtime_error &e) {
    std::cerr << "ERROR::" << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  bg_sub.LearnBGModel(cap);

  // 2º - Learn Skin Model using samples from camera or file
  if (argc >= 2) {
    std::ifstream means_file;
    try {
      means_file.open(argv[1]);
    } catch (const std::ios::ios_base::failure &e) {
      std::cerr << "ERROR::" << e.what() << std::endl;
      return EXIT_FAILURE;
    }

    bg_sub.LearnModel(means_file);
  } else {
    bg_sub.LearnModel(cap);
  }

  // 3º - Create windows
  const std::string reconocimiento = "Reconocimiento";
  const std::string fondo = "Fondo";
  cv::namedWindow(reconocimiento);
  cv::moveWindow(reconocimiento, 100, 50);
  cv::namedWindow(fondo);
  cv::moveWindow(fondo, 750, 50);

  int dilation_size = 3;
  cv::createTrackbar("Dilation size:", fondo, &dilation_size, 40, nullptr);

  int median_size = 11;
  cv::createTrackbar("Median size:", fondo, &median_size, 40,
                     correct_median_size, &median_size);

  // MAIN LOOP
  while (!quit) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
      std::cerr << "Read empty frame." << std::endl;
      continue;
    }

    // 4º - Background subtraction
    cv::Mat bgmask;
    bg_sub.ObtainBGMask(frame, bgmask);

    // 5º - Noise reduction
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {2 * dilation_size + 1, 2 * dilation_size + 1});

    cv::medianBlur(bgmask, bgmask, median_size);
    cv::morphologyEx(bgmask, bgmask, cv::MORPH_OPEN, element);
    cv::dilate(bgmask, bgmask, element);

    // 6º - Features detection
    hand_detector.FeaturesDetection(bgmask, frame);

    // 7º - Display results
    /* cv::flip(frame, frame, 1); */
    cv::imshow(reconocimiento, frame);
    cv::flip(bgmask, bgmask, 1);
    cv::imshow(fondo, bgmask);

    handle_input(cv::waitKey(40));
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
  // Median size must be always even
  int &median_size = *reinterpret_cast<int *>(userdata);
  if (median_size % 2 == 0)
    ++median_size;
}

void handle_input(int c) {
  switch (c) {
  case 27:  // Escape key
  case 'q': // Exit app
    quit = true;
    break;

  case 'r': // Relearn samples
    bg_sub.LearnModel(cap);
    break;

  case 'b': // Relearn background
    bg_sub.LearnBGModel(cap);
    break;

  case 't':
    bg_sub.ToggleBGMask();
    break;

  case 'd':
    hand_detector.ToggleDebugLines();
    break;
  }
}
