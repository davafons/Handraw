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
  void LearnModel(cv::VideoCapture &cap);    // From camera
  void LearnModel(std::istream &means_file); // From file

  // BG Model
  void LearnBGModel(cv::VideoCapture &cap);
  void ObtainBGMask(const cv::Mat &frame, cv::Mat &bgmask) const;

  // Toggle components of the BG subtractor
  void ToggleBGSubtractor() {
    bg_subtractor_enabled_ = !bg_subtractor_enabled_;
  }
  void ToggleFaceSubtractor() {
    face_subtractor_enabled_ = !face_subtractor_enabled_;
  };

private:
  // Thresholds for skin color detection
  int h_low_{18}, h_up_{17};
  int l_low_{13}, l_up_{73};
  int s_low_{31}, s_up_{9};
  std::vector<cv::Scalar> means_;

  // Settings of samples used in skin model learning
  int max_horiz_samples_{3};
  int max_vert_samples_{6};
  int distance_between_samples_{30};
  int sample_size_{20};

  // Subtractor for removing background before skin detection
  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_;
  int bg_samples_{30};
  bool bg_subtractor_enabled_{true};

  // Haar cascade classifier for face subtraction (SLOW!)
  mutable cv::CascadeClassifier face_subtractor_;
  bool face_subtractor_enabled_;

  // Windows created by this class
  const std::string win_trackbars_{"Trackbars"};

private:
  void RemoveBG(const cv::Mat &frame, cv::Mat &masked_frame) const;
  void RemoveFace(const cv::Mat &frame, cv::Mat &bgmask) const;

  void clamp(cv::Scalar &s, int low = 0, int up = 255) const;
};
