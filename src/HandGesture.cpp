#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "HandGesture.h"

HandGesture::HandGesture() {
  cv::namedWindow(win_gest_trackbars);
  cv::createTrackbar("Scale percentage", win_gest_trackbars, &scale_percentage_,
                     100);
  cv::createTrackbar("Min defect angle", win_gest_trackbars, &min_defect_angle_,
                     270);
  cv::createTrackbar("Max defect angle", win_gest_trackbars, &max_defect_angle_,
                     270);

  ok_image = cv::imread("res/ok1.jpg");
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
  std::vector<cv::Point> max_contour = contours[0];
  for (size_t i = 1; i < contours.size(); ++i) {
    if (contours[i].size() > max_contour.size()) {
      max_contour = contours[i];
      max_contour_index = i;
    }
  }

  if (debug_lines_) {
    cv::drawContours(output_img, contours, max_contour_index,
                     cv::Scalar(255, 0, 0), 2);
  }

  // Obtenemos el convex hull
  std::vector<int> hull_ints; // Para calcular los defectos de convexidad
  cv::convexHull(max_contour, hull_ints);

  std::vector<cv::Point> hull_points; // Para dibujar y el bounding rect
  cv::convexHull(max_contour, hull_points);

  // Calculamos el bounding rect (Para operaciones invariables a escala)
  cv::Rect hand_rect = cv::boundingRect(hull_points);

  // El centro del bounding rect lo guardamos para trackear el movimiento de la
  // mano
  cv::Point hand_rect_center = cv::Point(hand_rect.x + hand_rect.width / 2,
                                         hand_rect.y + hand_rect.height / 2);

  hand_points_[hand_points_index_ % hand_points_.size()] = hand_rect_center;
  ++hand_points_index_;

  // Pintamos el convex hull y el bounding rect
  if (debug_lines_) {
    cv::polylines(output_img, hull_points, true, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(output_img, hand_rect, cv::Scalar(0, 255, 255), 2);
  }

  // Obtenemos los defectos de convexidad
  std::vector<cv::Vec4i> defects;
  cv::convexityDefects(max_contour, hull_ints, defects);

  // Detectamos las puntas de los dedos a partir de los defectos
  finger_tips_.clear();
  finger_count_ = 0;

  std::vector<cv::Vec4i> filtered_defects;
  for (const auto &defect : defects) {
    if (finger_tips_.size() >= 5)
      break;

    cv::Point s = max_contour[defect[0]];
    cv::Point e = max_contour[defect[1]];
    cv::Point f = max_contour[defect[2]];

    float defect_depth = float(defect[3]) / 256.0;
    double angle = getAngle(s, e, f);

    if (defect_depth < hand_rect.height * float(scale_percentage_) / 100)
      continue;

    // Filtrar según el ángulo mínimo y máximo
    if (angle < min_defect_angle_ || angle > max_defect_angle_)
      continue;

    if (debug_lines_)
      cv::circle(output_img, f, 3, cv::Scalar(0, 0, 255), 2);

    filtered_defects.push_back(defect);
    finger_tips_.push_back(e);
  }

  finger_count_ = finger_tips_.size() + 1;

  if (finger_tips_.size() == 0) {
    int hand_rect_ratio = std::abs(hand_rect.width - hand_rect.height);
    if (hand_rect_ratio < 80)
      finger_count_ = 0;
  } else if (finger_tips_.size() == 1) {
    if (hand_rect.width > hand_rect.height)
      finger_count_ = 1;
  }
  for (const auto &point : finger_tips_)
    cv::circle(output_img, point, 5, cv::Scalar(255, 255, 0), -1);

  message_ = "";

  if (finger_count_ == 3) {
    cv::Point s = max_contour[filtered_defects[1][0]];
    cv::Point e = max_contour[filtered_defects[1][1]];

    if (cv::norm(s - e) > hand_rect.width * 0.56)
      message_ = "Rock!";
  } else if (finger_count_ == 2) {
    cv::Point s = max_contour[filtered_defects[0][0]];
    cv::Point e = max_contour[filtered_defects[0][1]];
    cv::Point f = max_contour[filtered_defects[0][2]];
    double angle = getAngle(s, e, f);
    std::cout << angle << std::endl;

    if (angle > 90 && angle < 120)
      message_ = "Loser!";
    else if (angle > 20 && angle < 60)
      message_ = "Peace!";

  } else if (finger_count_ == 4) {
    cv::Point s = max_contour[filtered_defects[0][0]];
    cv::Point e = max_contour[filtered_defects[0][1]];
    cv::Point f = max_contour[filtered_defects[0][2]];
    double angle = getAngle(s, e, f);

    cv::Point f2 = max_contour[filtered_defects[1][2]];
    int dist_f1_f2 = cv::norm(f - f2);

    if (angle > 50 && cv::abs(s.y - e.y) > hand_rect.height * 0.25 &&
        dist_f1_f2 < hand_rect.height * 0.2) {
      message_ = "OK";
    }
  }
}

void HandGesture::FingerDrawing(cv::Mat &output_img) {
  // Dibujar zonas de pincel
  cv::rectangle(output_img, red_rect, cv::Scalar(0, 0, 255), cv::FILLED);
  cv::rectangle(output_img, green_rect, cv::Scalar(0, 255, 0), cv::FILLED);
  cv::rectangle(output_img, blue_rect, cv::Scalar(255, 0, 0), cv::FILLED);
  cv::rectangle(output_img, clear_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::circle(output_img, cv::Point(580, 40), 30, drawing_color_, cv::FILLED);

  if (finger_count_ >= 2) {

    cv::Scalar new_color(0);

    for (const auto &tip : finger_tips_) {
      if (red_rect.contains(tip))
        new_color += cv::Scalar(0, 0, 255);
      if (green_rect.contains(tip))
        new_color += cv::Scalar(0, 255, 0);
      if (blue_rect.contains(tip))
        new_color += cv::Scalar(255, 0, 0);
      if (clear_rect.contains(tip)) {
        drawn_lines_.clear();
        current_line_.clear();
      }
    }

    if (new_color != cv::Scalar(0, 0, 0)) {
      if (!current_line_.empty()) {
        drawn_lines_.push_back(std::make_pair(current_line_, drawing_color_));
        current_line_.clear();
      }

      drawing_color_ = new_color;
    }
  }

  if (finger_count_ == 1)
    current_line_.push_back(cv::Point(finger_tips_.front()));

  // Draw lines
  cv::polylines(output_img, current_line_, false, drawing_color_, 2);

  for (const auto &line_color : drawn_lines_)
    cv::polylines(output_img, line_color.first, false, line_color.second, 2);
}

void HandGesture::DetectHandMovement(cv::Mat &output_img) {
  for (size_t i = 0; i < hand_points_.size() - 1; ++i) {
    cv::line(output_img,
             hand_points_[(i + hand_points_index_) % hand_points_.size()],
             hand_points_[(i + hand_points_index_ + 1) % hand_points_.size()],
             cv::Scalar(0, 0, 255),
             11 - (10.0f / hand_points_.size()) * (hand_points_.size() - i));
  }

  hand_direction_.clear();

  int dX = hand_points_[hand_points_index_ % hand_points_.size()].x -
           hand_points_[(hand_points_index_ - 1) % hand_points_.size()].x;
  int dY = hand_points_[hand_points_index_ % hand_points_.size()].y -
           hand_points_[(hand_points_index_ - 1) % hand_points_.size()].y;

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
  /*     std::abs(hand_points_[0].x - hand_points_[hand_points_.size() - 1].x);
   */
  /* int start_mid_diff = */
  /*     std::abs(hand_points_[0].x - hand_points_[hand_points_.size() / 2].x);
   */

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
