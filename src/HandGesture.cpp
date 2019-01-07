#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "HandGesture.h"

HandGesture::HandGesture() {}

void HandGesture::FeaturesDetection(cv::Mat mask, cv::Mat output_img) {

  std::vector<std::vector<cv::Point>> contours;
  cv::Mat temp_mask;
  mask.copyTo(temp_mask);

  // CODIGO 3.1
  // Detección del contorno de la mano y selección del contorno más largo
  cv::findContours(temp_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    std::cerr << "No contour detected!\n";
    return;
  }

  auto max_contour_iter =
      std::max_element(contours.cbegin(), contours.cend(),
                       [](std::vector<cv::Point> i, std::vector<cv::Point> j) {
                         return i.size() < j.size();
                       });

  int max_contour_index = max_contour_iter - contours.cbegin();
  auto max_contour = *max_contour_iter;

  // Pintar el contorno
  if (debug_lines_)
    cv::drawContours(output_img, contours, max_contour_index,
                     cv::Scalar(255, 0, 0), 2);

  // Obtenemos el convex hull
  std::vector<cv::Point> hull_points; // Para dibujar
  cv::convexHull(max_contour, hull_points);

  std::vector<int> hull_ints; // Para calcular los defectos de convexidad
  cv::convexHull(max_contour, hull_ints);

  // Pintamos el convex hull
  if (debug_lines_)
    cv::polylines(output_img, hull_points, true, cv::Scalar(0, 0, 255), 2);

  // Obtenemos los defectos de convexidad
  std::vector<cv::Vec4i> defects;
  cv::convexityDefects(max_contour, hull_ints, defects);

  cv::createTrackbar("Min depth", "Reconocimiento", &min_depth_, 200);
  cv::createTrackbar("Min length", "Reconocimiento", &min_length_, 400);
  cv::createTrackbar("Max length", "Reconocimiento", &max_length_, 400);
  cv::createTrackbar("Max neighbour distance", "Reconocimiento",
                     &max_neighbour_distance, 300);

  std::vector<cv::Point> good_points;
  for (int i = 0; i < defects.size(); ++i) {
    const cv::Vec4i &defect = defects[i];

    cv::Point s = max_contour[defect[0]];
    cv::Point e = max_contour[defect[1]];
    cv::Point f = max_contour[defect[2]];

    float depth = float(defect[3]) / 256.0;
    double angle = getAngle(s, e, f);

    // CODIGO 3.2
    // Filtrar y mostrar los defectos de convexidad
    if (depth < min_depth_)
      continue;

    if (debug_lines_)
      cv::circle(output_img, f, 3, cv::Scalar(0, 255, 255), 2);

    good_points.push_back(f);
  }


  int fingers = 0;
  // Get fingers
  if (!good_points.empty() && good_points.size() >= 3) {
    cv::Point2f min_center;
    float rad;
    cv::minEnclosingCircle(good_points, min_center, rad);

    if (debug_lines_) {
      cv::circle(output_img, min_center, rad, cv::Scalar(0, 255, 0), 2);
      cv::circle(output_img, min_center, 3, cv::Scalar(0, 255, 0), 2);
    }

    std::vector<cv::Point> merged_points = mergeNearPoints(hull_points);

    // Count fingers
    for (const auto &point : merged_points) {
      if (fingers == 5)
        break;

      if (point.y - 70 > min_center.y)
        continue;

      float diff = cv::norm(min_center - cv::Point2f(point));
      if (diff < min_length_ || diff > max_length_)
        continue;

      ++fingers;
      if (debug_lines_)
        cv::line(output_img, point, min_center, cv::Scalar(0, 0, 255), 2);

      cv::circle(output_img, point, 5, cv::Scalar(0, 255, 255), 3);
    }
  }

  cv::flip(output_img, output_img, 1);
  cv::putText(output_img, std::to_string(fingers), {150, 150},
              cv::FONT_HERSHEY_PLAIN, 8, cv::Scalar(255, 255, 255), 8);
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

std::vector<cv::Point>
HandGesture::mergeNearPoints(const std::vector<cv::Point> &points) const {
  std::vector<cv::Point> merged_points;

  std::vector<int> labels;
  cv::partition(points, labels, [this](const cv::Point &a, const cv::Point &b) {
    return cv::norm(a - b) < max_neighbour_distance;
  });

  int current_label = labels.front();
  std::vector<cv::Point> acc_points;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (current_label != labels[i]) {
      cv::Point total =
          std::accumulate(acc_points.cbegin(), acc_points.cend(), cv::Point(0));
      cv::Point mean_point(total.x / acc_points.size(),
                           total.y / acc_points.size());
      merged_points.push_back(mean_point);
      acc_points.clear();

      std::cout << current_label << " - " << mean_point << "\n";
      current_label = labels[i];
    }
    acc_points.push_back(points[i]);
  }

  /* if (!acc_points.empty()) { */
  /*   cv::Point total = */
  /*       std::accumulate(acc_points.cbegin(), acc_points.cend(),
   * cv::Point(0)); */
  /*   cv::Point mean_point(total.x / acc_points.size(), */
  /*                        total.y / acc_points.size()); */
  /*   merged_points.push_back(mean_point); */
  /* } */

  std::cout << "--" << std::endl;

  return merged_points;
}
