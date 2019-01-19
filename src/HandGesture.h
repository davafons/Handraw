#pragma once

#include <vector>
#include <deque>
#include <string>

#include <opencv2/opencv.hpp>
class HandGesture {
public:
  HandGesture();
  // Detección y filtrado de convexity defects
  void FeaturesDetection(const cv::Mat &mask, cv::Mat &output_img);

  // Funcionalidades extra de la aplicación (Dibujar, gestos...)
  void FingerDrawing(cv::Mat &output_img);
  void DetectHandGestures();
  void DetectHandMovement(cv::Mat &output_img);

  void ToggleDebugLines() { debug_lines_ = !debug_lines_; };
  void ToggleContourOrientation() { contour_oritentation_clockwise_ = !contour_oritentation_clockwise_; }
  void ToggleColorsPosition() { colors_position_top_ = !colors_position_top_; }

  int getFingerCount() const { return finger_count_; }
  std::string getHandDirection() const { return hand_directions_.back(); }
  std::string getMessage() const { return message_; }

private:
  // Valores para filtar los convexity defects
  int scale_percentage_{25};
  int min_defect_angle_{15};
  int max_defect_angle_{120};

  bool contour_oritentation_clockwise_{true};

  // Features detection
  std::vector<cv::Point> max_contour_;
  std::vector<cv::Vec4i> filtered_defects_;
  int finger_count_ = 0;
  cv::Rect hand_rect_;

  // Mostrar bounding rect/contour/convexity defects...
  bool debug_lines_{true};

  // Dibujo
  cv::Scalar drawing_color_{255, 0, 0};
  bool colors_position_top_{true};
  std::vector<cv::Point> current_line_;
  std::vector<std::pair<std::vector<cv::Point>, cv::Scalar>> drawn_lines_;

  // Movimiento de la mano
  std::deque<cv::Point> hand_points_{10, cv::Point(0)};
  std::deque<std::string> hand_directions_{10, ""};
  std::string message_{""};

  // Window
  const std::string w_gest_trackbars_{"Gesture Trackbacks"};

private:
  double getAngle(cv::Point s, cv::Point e, cv::Point f);
};
