#pragma once

#include <queue>
#include <string>

#include <opencv2/opencv.hpp>
class HandGesture {
public:
  HandGesture();
  void FeaturesDetection(const cv::Mat &mask, cv::Mat &output_img);
  void FingerDrawing(cv::Mat &output_img);
  void DetectHandGestures();
  void DetectHandMovement(cv::Mat &output_img);

  void ToggleDebugLines() { debug_lines_ = !debug_lines_; };

  int getFingerCount() const { return finger_count_; }
  std::string getHandDirection() const { return hand_direction_; }
  std::string getMessage() const { return message_; }

private:
  // Valores para filtar los convexity defects
  int scale_percentage_{25};
  int min_defect_angle_{15};
  int max_defect_angle_{120};

  // Fingers detection
  std::vector<cv::Point> max_contour_;
  cv::Rect hand_rect_;

  int finger_count_ = 0;
  std::vector<cv::Point> finger_tips_;
  std::vector<cv::Vec4i> filtered_defects_;

  // Mostrar bounding rect/contour/convexity defects...
  bool debug_lines_{true};

  // Objetos utilizados para dibujo
  cv::Scalar drawing_color_{255, 0, 0};
  cv::Rect red_rect{420, 400, 80, 80};
  cv::Rect blue_rect{340, 400, 80, 80};
  cv::Rect green_rect{260, 400, 80, 80};
  cv::Rect clear_rect{0, 400, 80, 80};
  std::vector<cv::Point> current_line_;
  std::vector<std::pair<std::vector<cv::Point>, cv::Scalar>> drawn_lines_;

  // Motion detection
  std::vector<cv::Point> hand_points_{10, cv::Point(0)};
  int hand_points_index_{0};
  std::string hand_direction_{""};
  std::string message_{""};

  // Window
  const std::string w_gest_trackbars_{"Gesture Trackbacks"};

private:
  double getAngle(cv::Point s, cv::Point e, cv::Point f);
};
