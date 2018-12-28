#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class MyBGSubtractorColor {
public:
  MyBGSubtractorColor();
  ~MyBGSubtractorColor();

  void LearnModel(cv::VideoCapture &cap);
  void ObtainBGMask(cv::Mat frame, cv::Mat &bgmask);


private:
  int h_low_{54}, h_up_{3};
  int l_low_{71}, l_up_{54};
  int s_low_{28}, s_up_{83};

  int max_horiz_samples_{2};
  int max_vert_samples_{5};
  const int max_samples_{max_horiz_samples_ * max_vert_samples_};

  int distance_between_samples_{30};

  int sample_size_{30};

  std::vector<cv::Scalar> lower_bounds_;
  std::vector<cv::Scalar> upper_bounds_;
  std::vector<cv::Scalar> means_;

  cv::Mat bgStatic_;
  cv::Ptr<cv::BackgroundSubtractor> pBgSub_;

  const std::string window_trackbars_{"Trackbars"};

private:
  void clamp(cv::Scalar &s, int low = 0, int up = 255);
};
