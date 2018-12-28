#include "MyBGSubtractorColor.h"

MyBGSubtractorColor::MyBGSubtractorColor() {
  cv::namedWindow(window_trackbars_);
  cv::namedWindow("test");

  lower_bounds_ = std::vector<cv::Scalar>(max_samples_);
  upper_bounds_ = std::vector<cv::Scalar>(max_samples_);

  cv::createTrackbar("H low:", window_trackbars_, &h_low_, 100, nullptr);
  cv::createTrackbar("H up:", window_trackbars_, &h_up_, 100, nullptr);
  cv::createTrackbar("L low:", window_trackbars_, &l_low_, 100, nullptr);
  cv::createTrackbar("L up:", window_trackbars_, &l_up_, 100, nullptr);
  cv::createTrackbar("S low:", window_trackbars_, &s_low_, 100, nullptr);
  cv::createTrackbar("S up:", window_trackbars_, &s_up_, 100, nullptr);
}

MyBGSubtractorColor::~MyBGSubtractorColor() {
  cv::destroyWindow(window_trackbars_);
  cv::destroyWindow("test");
}

void MyBGSubtractorColor::LearnModel(cv::VideoCapture &cap) {
  cv::Mat frame, hls_frame;
  cap >> frame;
  frame.copyTo(bgStatic_);

  // Get first frame as BG for subtraction
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  // Create window
  const std::string window_samples =
      "Cubre los cuadrados con la mano y pulsa espacio";
  cv::namedWindow(window_samples);
  cv::createTrackbar("Sample Size:", window_samples, &sample_size_, 100,
                     nullptr);
  cv::createTrackbar("Dist between Samples", window_samples,
                     &distance_between_samples_, 100, nullptr);
  cv::createTrackbar("Max horiz samples", window_samples, &max_horiz_samples_,
                     10, nullptr);
  cv::createTrackbar("Max vert samples", window_samples, &max_vert_samples_, 10,
                     nullptr);

  std::vector<cv::Point> samples_positions;

  // Loop while sampling hand
  while (cv::waitKey(40) != ' ') {
    /* flip(frame, frame, 1); */
    cv::Mat tmp_frame;
    frame.copyTo(tmp_frame);

    // Calculate and render sample rect positions
    samples_positions.clear();
    for (int i = 0; i < max_horiz_samples_; ++i) {
      for (int j = 0; j < max_vert_samples_; ++j) {
        cv::Point p;
        p.x = frame.cols / 2 + (-max_horiz_samples_ / 2 + i) *
                                   (sample_size_ + distance_between_samples_);
        p.y = frame.rows / 2 + (-max_vert_samples_ / 2 + j) *
                                   (sample_size_ + distance_between_samples_);
        samples_positions.push_back(p);

        cv::rectangle(tmp_frame,
                      cv::Rect(p.x, p.y, sample_size_, sample_size_),
                      cv::Scalar(0, 255, 0), 2);
      }
    }

    cv::imshow(window_samples, tmp_frame);
    cap >> frame;
  }

  cv::destroyWindow(window_samples);

  // CODIGO 1.1
  // Obtener las regiones de interés y calcular la media de cada una de ellas
  // almacenar las medias en la variable means_

  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  for (const auto &sample : samples_positions) {
    cv::Mat roi =
        hls_frame(cv::Rect(sample.x, sample.y, sample_size_, sample_size_));
    means_.push_back(cv::mean(roi));
  }
}

void MyBGSubtractorColor::ObtainBGMask(cv::Mat frame, cv::Mat &bgmask) {
  cv::Mat acc(frame.rows, frame.cols, CV_8U);
  acc.setTo(cv::Scalar(0));

  cv::Mat hls_frame;
  cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  for (const auto &mean : means_) {
    cv::Scalar low_bound(mean[0] - h_low_, mean[1] - s_low_, mean[2] - l_low_);
    cv::Scalar up_bound(mean[0] + h_up_, mean[1] + s_up_, +mean[2] + l_up_);

    clamp(low_bound, 0, 255);
    clamp(up_bound, 0, 255);

    cv::Mat temp_bgmask;
    cv::inRange(hls_frame, low_bound, up_bound, temp_bgmask);
    acc += temp_bgmask;
  }

  acc.copyTo(bgmask);
}

void MyBGSubtractorColor::clamp(cv::Scalar &s, int low, int up) {
  s[0] = (s[0] < low) ? low : (s[0] > up) ? up : s[0];
  s[1] = (s[1] < low) ? low : (s[1] > up) ? up : s[1];
  s[2] = (s[2] < low) ? low : (s[2] > up) ? up : s[2];
}
