#pragma once

#include <istream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class MyBGSubtractorColor {
public:
  MyBGSubtractorColor();
  ~MyBGSubtractorColor();

  void LearnModel(cv::VideoCapture &cap);
  void LearnModel(std::istream &means_file);
  void ObtainBGMask(cv::Mat frame, cv::Mat &bgmask) const;

private:
  int h_low_{61}, h_up_{22};
  int l_low_{30}, l_up_{55};
  int s_low_{71}, s_up_{19};

  int max_horiz_samples_{3};
  int max_vert_samples_{6};
  const int max_samples_{max_horiz_samples_ * max_vert_samples_};

  int distance_between_samples_{30};
  int sample_size_{20};

  std::vector<cv::Scalar> means_;

  const std::string win_trackbars_{"Trackbars"};

private:
  void clamp(cv::Scalar &s, int low = 0, int up = 255) const;
};
