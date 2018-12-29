#include <vector>

#include "HandGesture.h"

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

void HandGesture::FeaturesDetection(cv::Mat mask, cv::Mat output_img) {

  std::vector<std::vector<cv::Point>> contours;
  cv::Mat temp_mask;
  mask.copyTo(temp_mask);

  // CODIGO 3.1
  // Detección del contorno de la mano y selección del contorno más largo
  cv::findContours(temp_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  auto max_contour_iter =
      std::max_element(contours.cbegin(), contours.cend(),
                       [](std::vector<cv::Point> i, std::vector<cv::Point> j) {
                         return i.size() < j.size();
                       });

  int max_contour_index = max_contour_iter - contours.cbegin();
  auto max_contour = *max_contour_iter;

  // Pintar el contorno
  cv::drawContours(output_img, contours, max_contour_index,
                   cv::Scalar(255, 0, 0), 2);

  // Obtenemos el convex hull
  std::vector<cv::Point> hull_points; // Para dibujar
  cv::convexHull(max_contour, hull_points);

  std::vector<int> hull_ints; // Para calcular los defectos de convexidad
  cv::convexHull(max_contour, hull_ints);

  // Pintamos el convex hull
  cv::polylines(output_img, hull_points, true, cv::Scalar(0, 0, 255), 2);

  // Obtenemos los defectos de convexidad
  std::vector<cv::Vec4i> defects;
  cv::convexityDefects(max_contour, hull_ints, defects);

  cv::createTrackbar("Max angle", "Reconocimiento", &max_angle_, 180);


  for(int i = 0; i < defects.size(); ++i) {
    const cv::Vec4i& defect = defects[i];

    cv::Point s = max_contour[defect[0]];
    cv::Point e = max_contour[defect[1]];
    cv::Point f = max_contour[defect[2]];

    float depth = float(defect[3]) / 256.0;
    double angle = getAngle(s, e, f);

    // CODIGO 3.2
    // Filtrar y mostrar los defectos de convexidad
    if(angle > max_angle_)
      continue;

    cv::circle(output_img, hull_points[i], 5, cv::Scalar(0, 255, 255), 3);
    /* cv::circle(output_img, s, 5, cv::Scalar(255, 0, 0), 3); */
    /* cv::circle(output_img, e, 5, cv::Scalar(255, 255, 0), 3); */
  }
}
