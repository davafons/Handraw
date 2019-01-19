#pragma once

#include <istream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class MyBGSubtractorColor {
public:
  MyBGSubtractorColor();
  ~MyBGSubtractorColor();

  // Generar modelo de la mano
  void LearnModel(cv::VideoCapture &cap);    // Desde la camara
  void LearnModel(std::istream &means_file); // Desde un archivo

  // Crear máscara binaria
  void ObtainBGMask(const cv::Mat frame, cv::Mat &bgmask) const;

  // BG Sub
  void LearnBGModel(cv::VideoCapture &cap) const;
  void ToggleBGSubtractor() { bg_sub_enabled_ = !bg_sub_enabled_; }

private:
  // Rangos para la detección del color de la piel
  int h_low_{80}, h_up_{7};
  int l_low_{14}, l_up_{80};
  int s_low_{18}, s_up_{70};
  std::vector<cv::Scalar> means_;

  // Atributos de los Rect utilizados para calcular las ROI de la piel
  int max_horiz_samples_{3};
  int max_vert_samples_{6};
  int distance_between_samples_{30};
  int sample_size_{20};

  // Clase para realizar BG Sub
  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_sub_;
  int bg_samples_{20};
  bool bg_sub_enabled_{true};

  // Windows
  const std::string w_trackbars_{"Trackbars"};
  const std::string w_bg_{"Background Model"};
  const std::string w_bg_sub_{"BG Sub"};

private:
  void RemoveBG(const cv::Mat frame, cv::Mat &masked_frame) const;

  void clamp(cv::Scalar &s, int low = 0, int up = 255) const;
};
