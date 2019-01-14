#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "HandGesture.h"

HandGesture::HandGesture() {
  cv::namedWindow(w_gest_trackbars_);
  cv::createTrackbar("Scale percentage", w_gest_trackbars_, &scale_percentage_,
                     100);
  cv::createTrackbar("Min defect angle", w_gest_trackbars_, &min_defect_angle_,
                     270);
  cv::createTrackbar("Max defect angle", w_gest_trackbars_, &max_defect_angle_,
                     270);
}

void HandGesture::FeaturesDetection(const cv::Mat &mask, cv::Mat &output_img) {

  // CODIGO 3.1
  // Detección del contorno de la mano y selección del contorno más largo

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    std::cerr << "No contour detected!\n";
    return;
  }

  int max_contour_index = -1;
  max_contour_ = contours[0];
  for (size_t i = 1; i < contours.size(); ++i) {
    if (contours[i].size() > max_contour_.size()) {
      max_contour_ = contours[i];
      max_contour_index = i;
    }
  }

  // Obtenemos el convex hull
  std::vector<int> hull_ints; // Para calcular los defectos de convexidad
  cv::convexHull(max_contour_, hull_ints, contour_oritentation_clockwise_);

  std::vector<cv::Point> hull_points; // Para dibujar y el bounding rect
  cv::convexHull(max_contour_, hull_points, contour_oritentation_clockwise_);

  // Calculamos el bounding rect (Para operaciones invariables a escala)
  hand_rect_ = cv::boundingRect(hull_points);
  // El centro del bounding rect lo guardamos para trackear el movimiento de la
  // mano
  cv::Point hand_rect_center = cv::Point(hand_rect_.x + hand_rect_.width / 2,
                                         hand_rect_.y + hand_rect_.height / 2);

  hand_points_.push_back(hand_rect_center);
  hand_points_.pop_front();

  // Pintamos countour, convex hull y bounding rect
  if (debug_lines_) {
    cv::drawContours(output_img, contours, max_contour_index, {255, 0, 0}, 2);
    cv::polylines(output_img, hull_points, true, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(output_img, hand_rect_, cv::Scalar(0, 255, 255), 2);
  }

  // Obtenemos los defectos de convexidad
  std::vector<cv::Vec4i> defects;
  cv::convexityDefects(max_contour_, hull_ints, defects);

  finger_count_ = 0;
  // Guardamos los defectos filtrados para hacer cálculos con ellos
  filtered_defects_.clear();

  for (const auto &defect : defects) {
    if (filtered_defects_.size() >= 5)
      break;

    cv::Point s = max_contour_[defect[0]];
    cv::Point e = max_contour_[defect[1]];
    cv::Point f = max_contour_[defect[2]];

    float defect_depth = float(defect[3]) / 256.0;
    double angle = getAngle(s, e, f);

    // Filtar según porcentaje del bounding rect
    if (defect_depth < hand_rect_.height * float(scale_percentage_) / 100)
      continue;

    // Filtrar según el ángulo mínimo y máximo
    if (angle < min_defect_angle_ || angle > max_defect_angle_)
      continue;

    if (debug_lines_)
      cv::circle(output_img, f, 3, cv::Scalar(0, 0, 255), 2);

    filtered_defects_.push_back(defect);

    cv::circle(output_img, s, 5, cv::Scalar(255, 255, 0), -1);
  }

  finger_count_ = filtered_defects_.size() + 1;

  // Si no detecta ningún defecto y el bounding rect es más cuadrado, es un puño
  // cerrado
  if (filtered_defects_.size() == 0) {
    int hand_ratio = std::abs(hand_rect_.width - hand_rect_.height);
    if (hand_ratio < 80)
      finger_count_ = 0;

    // Si solo detecta un defecto y es más ancho que largo, es el pulgar
  } else if (filtered_defects_.size() == 1) {
    if (hand_rect_.width > hand_rect_.height)
      finger_count_ = 1;
  }
}

void HandGesture::FingerDrawing(cv::Mat &output_img) {
  // Dibujar zonas de pincel
  cv::Rect red_rect{420, 380, 80, 80};
  cv::Rect blue_rect{340, 380, 80, 80};
  cv::Rect green_rect{260, 380, 80, 80};
  cv::Rect clear_rect{0, 380, 80, 80};

  if (colors_position_top_)
    red_rect.y = blue_rect.y = green_rect.y = clear_rect.y = 0;

  cv::rectangle(output_img, red_rect, {0, 0, 255}, cv::FILLED);
  cv::rectangle(output_img, green_rect, {0, 255, 0}, cv::FILLED);
  cv::rectangle(output_img, blue_rect, {255, 0, 0}, cv::FILLED);
  cv::rectangle(output_img, clear_rect, {255, 255, 255}, cv::FILLED);
  cv::circle(output_img, {580, 40}, 30, drawing_color_, cv::FILLED);

  if (finger_count_ >= 2) {

    cv::Scalar new_color(0);

    for (const auto &defect : filtered_defects_) {
      if (red_rect.contains(max_contour_[defect[0]]))
        new_color += cv::Scalar(0, 0, 255);
      if (green_rect.contains(max_contour_[defect[0]]))
        new_color += cv::Scalar(0, 255, 0);
      if (blue_rect.contains(max_contour_[defect[0]]))
        new_color += cv::Scalar(255, 0, 0);
      if (clear_rect.contains(max_contour_[defect[0]])) {
        drawn_lines_.clear();
        current_line_.clear();
      }
    }

    if (new_color != cv::Scalar(0, 0, 0))
      drawing_color_ = new_color;

    if (finger_count_ == 2)
      current_line_.push_back(max_contour_[filtered_defects_[0][0]]);
    else if (!current_line_.empty()) {
      drawn_lines_.push_back(std::make_pair(current_line_, drawing_color_));
      current_line_.clear();
    }
  }

  // Draw lines
  cv::polylines(output_img, current_line_, false, drawing_color_, 2);

  for (const auto &line_color : drawn_lines_)
    cv::polylines(output_img, line_color.first, false, line_color.second, 2);
}

void HandGesture::DetectHandGestures() {
  message_ = "";

  if (finger_count_ >= 2) {
    cv::Point s1 = max_contour_[filtered_defects_[0][0]];
    cv::Point e1 = max_contour_[filtered_defects_[0][1]];
    cv::Point f1 = max_contour_[filtered_defects_[0][2]];
    double angle1 = getAngle(s1, e1, f1);

    // Gestos con 2 dedos:
    if (finger_count_ == 2) {

      if (angle1 > 80 && angle1 < 120)
        message_ = "Loser!";
      else if (angle1 > 20 && angle1 < 60)
        message_ = "Peace!";
    }

    if (finger_count_ >= 3) {
      cv::Point s2 = max_contour_[filtered_defects_[1][0]];
      cv::Point e2 = max_contour_[filtered_defects_[1][1]];
      cv::Point f2 = max_contour_[filtered_defects_[1][2]];

      // Gestos con 3 dedos:
      if (finger_count_ == 3) {

        int index_pinky_dist = cv::norm(s2 - e2);

        if (index_pinky_dist > hand_rect_.width * 0.56)
          message_ = "Rock!";
      }

      // Gestos con 4 dedos:
      if (finger_count_ == 4) {

        int dist_f1_f2 = cv::norm(f1 - f2);
        int vertical_dist = cv::abs(s1.y - e1.y);

        if (angle1 > 50 && vertical_dist > hand_rect_.height * 0.25 &&
            dist_f1_f2 < hand_rect_.height * 0.2) {
          message_ = "OK";
        }
      }
    }
  }
}

void HandGesture::DetectHandMovement(cv::Mat &output_img) {
  for (size_t i = 0; i < hand_points_.size() - 1; ++i) {
    cv::line(output_img, hand_points_[i], hand_points_[i + 1],
             cv::Scalar(0, 0, 255),
             11 - (10.0f / hand_points_.size()) * (hand_points_.size() - i));
  }

  hand_direction_.clear();

  int dX = hand_points_.front().x - hand_points_.back().x;
  int dY = hand_points_.front().y - hand_points_.back().y;

  if (std::abs(dX) > 40) {
    if (dX > 0)
      hand_direction_ += "Derecha";
    else
      hand_direction_ += "Izquierda";
  }

  if (std::abs(dY) > 40) {
    if (dY > 0)
      hand_direction_ += "Arriba";
    else
      hand_direction_ += "Abajo";
  }

  if (hand_direction_.empty())
    hand_direction_ += "Quieta";

  /* int start_end_diff = */
  /*     std::abs(hand_points_[hand_points_index_ % hand_points_.size()].x - hand_points_[(hand_points_index_ - 1) % hand_points_.size()].x); */
  /* int start_mid_diff = */
  /*     std::abs(hand_points_[hand_points_index_ % hand_points_.size()].x - hand_points_[(hand_points_index_ % hand_points_.size()) / 2].x); */

  /* if(hand_points_index_ % hand_points_.size() == 0) */
  /*   std::cout << "--" << std::endl; */

  /* if (start_end_diff < 40 && start_mid_diff < 150) */
  /*   message_ = "Hola!"; */
  /* else */
  /*   message_ = ""; */

  /* std::cout << start_end_diff << " - " << start_mid_diff << std::endl; */
}

double HandGesture::getAngle(cv::Point s, cv::Point e, cv::Point f) {
  double v1[2], v2[2];
  v1[0] = s.x - f.x;
  v1[1] = s.y - f.y;

  v2[0] = e.x - f.x;
  v2[1] = e.y - f.y;

  double ang1 = atan2(v1[1], v1[0]);
  double ang2 = atan2(v2[1], v2[0]);

  double angle = ang1 - ang2;

  if (angle > CV_PI)
    angle -= 2 * CV_PI;

  if (angle < -CV_PI)
    angle += 2 * CV_PI;

  return (angle * 180.0 / CV_PI);
}
