#pragma once

#include <opencv2/opencv.hpp>

class HandGesture {
public:
  HandGesture() = default;
  void FeaturesDetection(cv::Mat mask, cv::Mat output_img);

private:
  double getAngle(cv::Point s, cv::Point e, cv::Point f);
};
