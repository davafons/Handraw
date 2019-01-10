#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class HandGesture {
public:
  HandGesture();
  void FeaturesDetection(cv::Mat mask, cv::Mat output_img);

  void ToggleDebugLines() { debug_lines_ = !debug_lines_; };

  int getFingerCount() const { return fingers_.size(); }

private:
  // Convexity defects filtering
  int min_depth_{25};
  int min_defect_angle_{20};
  int max_defect_angle_{180};

  // Fingers detection
  int max_neighbour_distance{33};
  int palm_threshold_{30};
  std::vector<cv::Point> fingers_;

  bool debug_lines_{false};

  bool use_alternative_method_{false};

  const std::string win_gest_trackbars{"Gesture Trackbacks"};

private:
  void DetectHandFingers(const std::vector<cv::Point> &contour, cv::Mat output_img);

  double getAngle(cv::Point s, cv::Point e, cv::Point f);
};
