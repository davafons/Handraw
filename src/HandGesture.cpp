#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "HandGesture.h"

HandGesture::HandGesture() {
  cv::namedWindow(win_gest_trackbars, cv::WINDOW_GUI_EXPANDED);
  cv::createTrackbar("Min depth", win_gest_trackbars, &min_depth_, 200);
  cv::createTrackbar("Max neighbour distance", win_gest_trackbars,
                     &max_neighbour_distance, 300);
  cv::createTrackbar("Palmn threshold", win_gest_trackbars, &palm_threshold_,
                     100);
  cv::createTrackbar("Min defect angle", win_gest_trackbars, &min_defect_angle_,
                     270);
  cv::createTrackbar("Max defect angle", win_gest_trackbars, &max_defect_angle_,
                     270);
}

void HandGesture::FeaturesDetection(cv::Mat mask, cv::Mat output_img) {
  fingers_.clear();
  std::vector<std::vector<cv::Point>> contours;

  // CODIGO 3.1
  // Detección del contorno de la mano y selección del contorno más largo

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

  if (debug_lines_)
    cv::drawContours(output_img, contours, max_contour_index,
                     cv::Scalar(255, 0, 0), 2);

  DetectHandFingers(max_contour, output_img);
}

void HandGesture::DetectHandFingers(const std::vector<cv::Point> &contour,
                                    cv::Mat output_img) {
  // Obtenemos el convex hull
  std::vector<cv::Point> hull_points; // Para dibujar y localizar los dedos
  cv::convexHull(contour, hull_points);

  std::vector<int> hull_ints; // Para calcular los defectos de convexidad
  cv::convexHull(contour, hull_ints);

  // Pintamos el convex hull
  if (debug_lines_)
    cv::polylines(output_img, hull_points, true, cv::Scalar(0, 0, 255), 2);

  // Obtenemos los defectos de convexidad
  std::vector<cv::Vec4i> defects;
  cv::convexityDefects(contour, hull_ints, defects);

    std::vector<cv::Point> tips;
    cv::Point last_point;
    for (const auto &defect : defects) {
      cv::Point s = contour[defect[0]];
      cv::Point e = contour[defect[1]];
      cv::Point f = contour[defect[2]];

      float defect_depth = float(defect[3]) / 256.0;
      double angle = getAngle(s, e, f);

      if (defect_depth < min_depth_)
        continue;

      if (angle < min_defect_angle_ || angle > max_defect_angle_)
        continue;

      if (debug_lines_)
        cv::circle(output_img, f, 3, cv::Scalar(0, 0, 255), 2);

      tips.push_back(s);
      last_point = e;
    }

    tips.push_back(last_point);

    cv::Rect hand_rect = cv::boundingRect(hull_points);
    cv::Point hand_rect_center(hand_rect.x + hand_rect.width / 2,
                               hand_rect.y + hand_rect.height / 2);

    if(debug_lines_) {
      cv::rectangle(output_img, hand_rect, cv::Scalar(0, 255, 255), 2);
      cv::circle(output_img, hand_rect_center, 3, cv::Scalar(0, 255, 255), -1);
    }

    for (const auto &point : tips)
      cv::circle(output_img, point, 5, cv::Scalar(255, 255, 0), -1);
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
