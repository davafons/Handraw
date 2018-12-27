#include "MyBGSubtractorColor.h"

MyBGSubtractorColor::MyBGSubtractorColor() {
  cv::namedWindow(window_trackbars_);

  lower_bounds_ = std::vector<cv::Scalar>(max_samples_);
  upper_bounds_ = std::vector<cv::Scalar>(max_samples_);
  means_ = std::vector<cv::Scalar>(max_samples_);

  cv::createTrackbar("H low:", window_trackbars_, &h_low_, 100, nullptr);
  cv::createTrackbar("H up:", window_trackbars_, &h_up_, 100, nullptr);
  cv::createTrackbar("L low:", window_trackbars_, &l_low_, 100, nullptr);
  cv::createTrackbar("L up:", window_trackbars_, &l_up_, 100, nullptr);
  cv::createTrackbar("S low:", window_trackbars_, &s_low_, 100, nullptr);
  cv::createTrackbar("S up:", window_trackbars_, &s_up_, 100, nullptr);
}

MyBGSubtractorColor::~MyBGSubtractorColor() {
  cv::destroyWindow(window_trackbars_);
}

void MyBGSubtractorColor::LearnModel(cv::VideoCapture &cap) {
  cv::Mat frame, hls_frame;
  cap >> frame;

  std::vector<cv::Point> samples_positions;

  for (unsigned i = 0; i < max_horiz_samples_; ++i) {
    for (unsigned j = 0; j < max_vert_samples_; ++j) {
      cv::Point p;
      p.x = frame.cols / 2 + (-max_horiz_samples_ / 2 + i) *
                                 (sample_size_ + distance_between_samples_);
      p.y = frame.rows / 2 + (-max_horiz_samples_ / 2 + j) *
                                 (sample_size_ + distance_between_samples_);

      samples_positions.push_back(p);
    }
  }

  // Create window
  const std::string window_samples =
      "Cubre los cuadrados con la mano y pulsa espacio";
  cv::namedWindow(window_samples);
  cv::createTrackbar("Sample Size:", window_samples, &sample_size_, 100,
                     nullptr);


  // Loop
  while (cv::waitKey(40) != ' ') {
    flip(frame, frame, 1);

    cv::Mat tmp_frame;
    frame.copyTo(tmp_frame);

    for (const auto &sample : samples_positions) {
      cv::rectangle(tmp_frame,
                    cv::Rect(sample.x, sample.y, sample_size_, sample_size_),
                    cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow(window_samples, tmp_frame);

    cap >> frame;
  }

  cv::destroyWindow(window_samples);
}

static void clamp(cv::Scalar &s, int low = 0, int up = 255) {
  s[0] = (s[0] < low) ? low : (s[0] > up) ? up : s[0];
  s[1] = (s[1] < low) ? low : (s[1] > up) ? up : s[1];
  s[2] = (s[2] < low) ? low : (s[2] > up) ? up : s[2];
}
