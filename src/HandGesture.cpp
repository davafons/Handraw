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
  int index = -1;

  // CODIGO 3.1
  // detección del contorno de la mano y selección del contorno más largo
  findContours(temp_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  int mmax_size = 0;
  for (int i = 0; i < contours.size(); ++i) {
    if (contours[i].size() >= mmax_size) {
      mmax_size = contours[i].size();
      index = i;
    }
  }

  // pintar el contorno
  drawContours(output_img, contours, index, cv::Scalar(255, 0, 0), 2, 8,
               std::vector<cv::Vec4i>(), 0, cv::Point());

  // obtener el convex hull
  std::vector<int> hull;
  convexHull(contours[index], hull);

  // pintar el convex hull
  cv::Point pt0 = contours[index][hull[hull.size() - 1]];
  for (int i = 0; i < hull.size(); i++) {
    cv::Point pt = contours[index][hull[i]];
    line(output_img, pt0, pt, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    pt0 = pt;
  }

  // obtener los defectos de convexidad
  std::vector<cv::Vec4i> defects;
  convexityDefects(contours[index], hull, defects);

  int cont = 0;
  for (int i = 0; i < defects.size(); i++) {
    cv::Point s = contours[index][defects[i][0]];
    cv::Point e = contours[index][defects[i][1]];
    cv::Point f = contours[index][defects[i][2]];
    float depth = (float)defects[i][3] / 256.0;
    double angle = getAngle(s, e, f);

    // CODIGO 3.2
    // filtrar y mostrar los defectos de convexidad
    //...
    /* if(angle < 10.0f || angle > 100.0f) */
    /*   continue; */

    if (depth < 80)
      continue;

    ++cont;
    circle(output_img, f, 5, cv::Scalar(0, 255, 0), 3);
  }

  putText(output_img, std::to_string(cont), cv::Point(130, 130),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 4, cv::Scalar(255, 50, 50), 1, cv::LINE_AA);
}
