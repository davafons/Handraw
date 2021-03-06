#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "HandGesture.h"
#include "MyBGSubtractorColor.h"

void open_camera(cv::VideoCapture &cap, unsigned int index);

void correct_median_size(int, void *);
void update_structuring_element(int, void *);

void handle_input(int c);
void show_help_message();

// -- Global variables --
const int MAX_EMPTY_FRAMES_TO_READ = 2000;

bool quit = false;
bool draw_enabled = false;

int dilation_size = 1;
int median_size = 11;
cv::Mat element = cv::getStructuringElement(
    cv::MORPH_ELLIPSE, {2 * dilation_size + 1, 2 * dilation_size + 1});

cv::VideoCapture cap;
MyBGSubtractorColor bg_sub;
HandGesture hand_detector;

int main(int argc, char *argv[]) {
  show_help_message();

  // 1º - Abrir cámara
  try {
    open_camera(cap, 0);
  } catch (const std::runtime_error &e) {
    std::cerr << "ERROR::" << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // 2º - Obtener una imagen estática del fondo (Para luego aplicar BG Sub)
  bg_sub.LearnBGModel(cap);

  // 3º - Obtener el modelo (color) de la mano para poder distinguirla
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

  // 4º - Crear ventanas para mostrar los resultados y trackbars
  const std::string reconocimiento = "Reconocimiento";
  const std::string fondo = "Fondo";
  cv::namedWindow(reconocimiento);
  cv::moveWindow(reconocimiento, 100, 50);
  cv::namedWindow(fondo);
  cv::moveWindow(fondo, 750, 50);

  cv::createTrackbar("Dilation size:", fondo, &dilation_size, 40,
                     update_structuring_element);
  cv::createTrackbar("Median size:", fondo, &median_size, 40,
                     correct_median_size);

  // MAIN LOOP
  while (!quit) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
      std::cerr << "Read empty frame." << std::endl;
      continue;
    }

    // 5º - Eliminar el fondo y obtener la máscara binaria
    cv::Mat bgmask;
    bg_sub.ObtainBGMask(frame, bgmask);

    // 6º - Reducir el ruido de la máscara binaria
    cv::morphologyEx(bgmask, bgmask, cv::MORPH_OPEN, element);
    cv::dilate(bgmask, bgmask, element);
    cv::medianBlur(bgmask, bgmask, median_size);

    // 7º - Detectar defectos de convexidad y dedos
    hand_detector.FeaturesDetection(bgmask, frame);

    // 7Aº - Dibujar
    if (draw_enabled)
      hand_detector.FingerDrawing(frame);

    // 7Bº - Detectar dirección de movimiento de la mano
    hand_detector.DetectHandMovement(frame);
    hand_detector.DetectHandGestures();

    // 8º - Mostrar los resultados
    cv::flip(frame, frame, 1);

    cv::putText(frame, std::to_string(hand_detector.getFingerCount()),
                {520, 75}, cv::FONT_HERSHEY_SIMPLEX, 2, {255, 255, 255}, 3);
    cv::putText(frame, hand_detector.getHandDirection(), {410, 460},
                cv::FONT_HERSHEY_SIMPLEX, 1, {0, 0, 255}, 2);
    cv::putText(frame, hand_detector.getMessage(), {430, 410},
                cv::FONT_HERSHEY_SIMPLEX, 2, {0, 0, 0}, 3);

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

void open_camera(cv::VideoCapture &cap, unsigned int index) {
  if (!cap.open(index))
    throw std::runtime_error("No se pudo abrir la cámara!");

  cv::Mat frame;
  int cont = 0;
  while (frame.empty() && ++cont < 2000)
    cap >> frame;

  if (cont >= MAX_EMPTY_FRAMES_TO_READ)
    throw std::runtime_error(
        "No se ha podido leer ningún frame valido tras iniciar la cámara!");
}

void correct_median_size(int, void *) {
  // Median size siempre debe ser impar
  if (median_size % 2 == 0)
    ++median_size;
}

void update_structuring_element(int, void *) {
  element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, {2 * dilation_size + 1, 2 * dilation_size + 1});
}

void handle_input(int c) {
  switch (c) {
  case 27:  // Escape key
  case 'q': // Salir de la app
    quit = true;
    break;

  case 'r': // Reaprender modelo
    bg_sub.LearnModel(cap);
    break;

  case 'b': // Reaprender fondo
    bg_sub.LearnBGModel(cap);
    break;

  case 't': // Activar BG Sub
    bg_sub.ToggleBGSubtractor();
    break;

  case 'd': // Mostrar las líneas de debug
    hand_detector.ToggleDebugLines();
    break;

  case 'k': // Activar dibujar
    draw_enabled = !draw_enabled;
    break;

  case 'l': // Cambiar la posición de la interfaz de dibujo
    hand_detector.ToggleColorsPosition();
    break;

  case 'c': // Crea el convex hull en el sentido de las agujas del reloj o no
    hand_detector.ToggleContourOrientation();
    break;
  }
}

void show_help_message() {
  std::cout << "Teclas utilizadas:" << std::endl;
  std::cout << "\t- [q] [Escape]   --> Salir de la aplicación" << std::endl;
  std::cout << "\t- [r]\t\t --> Reaprender modelo de la mano" << std::endl;
  std::cout << "\t- [b]\t\t --> Reaprender modelo de fondo" << std::endl;
  std::cout << "\t- [r]\t\t --> Activar/Desactivar BG Sub MOG2" << std::endl;
  std::cout << "\t- [d]\t\t --> Mostrar líneas de contorno, bounding rect, convex hull..." << std::endl;
  std::cout << "\t- [k]\t\t --> Activar/Desactivar interfaz para dibujar" << std::endl;
  std::cout << "\t- [l]\t\t --> Cambiar la posición de la interfaz de dibujo" << std::endl;
  std::cout << "\t- [c]\t\t --> Cambia el sentido en el que se guardan los puntos del convex hull (clockwise / counterclockwise) [DEFAULT: clockwise]" << std::endl;
}
