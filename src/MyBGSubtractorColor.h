#pragma once

#include <istream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class MyBGSubtractorColor {
public:
  MyBGSubtractorColor();
  ~MyBGSubtractorColor();

  // Skin Model
  void LearnModel(cv::VideoCapture &cap);
  void LearnModel(std::istream &means_file);

  // BG Model
  void LearnBGModel(cv::VideoCapture &cap);
  void RemoveBG(cv::Mat frame, cv::Mat &masked_frame);
  void ObtainBGMask(cv::Mat frame, cv::Mat &bgmask) const;
  void ToggleBGMask() { bg_subtractor_enabled_ = !bg_subtractor_enabled_; }

private:
  int h_low_{67}, h_up_{36};
  int l_low_{77}, l_up_{72};
  int s_low_{34}, s_up_{21};

  int max_horiz_samples_{3};
  int max_vert_samples_{6};
  const int max_samples_{max_horiz_samples_ * max_vert_samples_};

  int distance_between_samples_{30};
  int sample_size_{20};

  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_;
  int max_bg_samples_{20};
  bool bg_subtractor_enabled_{true};

  std::vector<cv::Scalar> means_;

  const std::string win_trackbars_{"Trackbars"};

private:
  void clamp(cv::Scalar &s, int low = 0, int up = 255) const;

  void RemoveBG(cv::Mat frame, cv::Mat &masked_frame) const;
};
