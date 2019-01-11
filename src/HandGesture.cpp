#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "HandGesture.h"

HandGesture::HandGesture() {
  cv::namedWindow(win_gest_trackbars);
  cv::createTrackbar("Min depth", win_gest_trackbars, &min_depth_, 200);
  cv::createTrackbar("Min defect angle", win_gest_trackbars, &min_defect_angle_,
                     270);
  cv::createTrackbar("Max defect angle", win_gest_trackbars, &max_defect_angle_,
                     270);
  cv::createTrackbar("Scale percentage", win_gest_trackbars, &scale_percentage_,
                     100);
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

  cv::Point last_point;
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

    finger_tips_.push_back(s);
    last_point = e;
  }

  for (const auto &point : finger_tips_)
    cv::circle(output_img, point, 5, cv::Scalar(255, 255, 0), -1);
}

void HandGesture::FingerDrawing(cv::Mat &output_img) {
  // Dibujar zonas de pincel
  cv::rectangle(output_img, red_rect, cv::Scalar(0, 0, 255), cv::FILLED);
  cv::rectangle(output_img, green_rect, cv::Scalar(0, 255, 0), cv::FILLED);
  cv::rectangle(output_img, blue_rect, cv::Scalar(255, 0, 0), cv::FILLED);
  cv::circle(output_img, cv::Point(580, 40), 30, drawing_color_, cv::FILLED);

  bool changing_color = false;
  if(finger_tips_.size() >= 1) {
    cv::Scalar new_color(0);
    for(const auto & tip : finger_tips_) {
      if(red_rect.contains(tip))
        new_color += cv::Scalar(0, 0, 255);
      if(green_rect.contains(tip))
        new_color += cv::Scalar(0, 255, 0);
      if(blue_rect.contains(tip))
        new_color += cv::Scalar(255, 0, 0);
    }
    std::cout << new_color << "\n";

    if(new_color != cv::Scalar(0, 0, 0)) {
      drawing_color_ = new_color;
      changing_color = true;

      if(!current_line_.empty()) {
        drawn_lines_.push_back(std::make_pair(current_line_, drawing_color_));
        current_line_.clear();
      }
    }
  }

  if(finger_tips_.size() == 1) {
    if(!changing_color) {
      current_line_.push_back(cv::Point(finger_tips_.front()));
    }
  }

  // Draw lines
  cv::polylines(output_img, current_line_, false, drawing_color_, 2);

  for(const auto & line_color : drawn_lines_)
    cv::polylines(output_img, line_color.first, false, line_color.second, 2);
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
