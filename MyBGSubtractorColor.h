#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class MyBGSubtractorColor {
public:
  MyBGSubtractorColor();
  ~MyBGSubtractorColor();

  void LearnModel(cv::VideoCapture &cap);

  static void clamp(cv::Scalar &s, int low = 0, int up = 255);

private:
  int h_low_{42}, h_up_{4};
  int l_low_{44}, l_up_{38};
  int s_low_{14}, s_up_{3};

  const unsigned max_horiz_samples_{3};
  const unsigned max_vert_samples_{6};
  const unsigned max_samples_{max_horiz_samples_ * max_vert_samples_};

  int distance_between_samples_{30};
  int sample_size_{20};

  std::vector<cv::Scalar> lower_bounds_;
  std::vector<cv::Scalar> upper_bounds_;
  std::vector<cv::Scalar> means_;

  const std::string window_trackbars_{"Trackbars"};
};
