#pragma once

#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>

class HandGesture {
public:
  HandGesture();
  void FeaturesDetection(const cv::Mat &mask, cv::Mat &output_img);
  void FingerDrawing(cv::Mat &output_img);

  void ToggleDebugLines() { debug_lines_ = !debug_lines_; };

  int getFingerCount() const { return finger_tips_.size(); }

private:
  // Convexity defects filtering
  int min_depth_{34};
  int min_defect_angle_{28};
  int max_defect_angle_{193};
  int scale_percentage_{15};

  // Fingers detection
  std::vector<cv::Point> finger_tips_;

  // Draw debug lines (contour, convexity defects, bounding rect)
  bool debug_lines_{false};

  // Drawing
  cv::Scalar drawing_color_{255, 0, 0};
  cv::Rect red_rect{100, 0, 80, 80};
  cv::Rect blue_rect{180, 0, 80, 80};
  cv::Rect green_rect{260, 0, 80, 80};
  std::vector<cv::Point> current_line_;
  std::vector<std::pair<std::vector<cv::Point>, cv::Scalar>> drawn_lines_;

  // Window
  const std::string win_gest_trackbars{"Gesture Trackbacks"};

private:
  double getAngle(cv::Point s, cv::Point e, cv::Point f);
};
