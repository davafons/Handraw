#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class HandGesture {
public:
  HandGesture();
  void FeaturesDetection(cv::Mat mask, cv::Mat output_img);

  void ToggleDebugLines() { debug_lines_ = !debug_lines_; };

  int getFingerCount() const { return fingers_; }

private:
  // Convexity defects filtering
  int min_depth_{7};
  int min_defect_angle_{20};
  int max_defect_angle_{180};

  // Fingers detection
  int max_neighbour_distance{33};
  int palm_threshold_{30};
  int fingers_{0};

  bool debug_lines_{false};

  const std::string win_gest_trackbars{"Gesture Trackbacks"};

private:
  double getAngle(cv::Point s, cv::Point e, cv::Point f);
  std::vector<cv::Point> mergeNearPoints(const std::vector<cv::Point> &points) const;
};
