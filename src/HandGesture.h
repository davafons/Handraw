#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class HandGesture {
public:
  HandGesture();
  void FeaturesDetection(cv::Mat mask, cv::Mat output_img);

private:
  int min_depth_{7};
  int min_length_{100};
  int max_length_{210};

  int max_neighbour_distance{40};

private:
  double getAngle(cv::Point s, cv::Point e, cv::Point f);
  std::vector<cv::Point> mergeNearPoints(const std::vector<cv::Point> &points) const;
};
