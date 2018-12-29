#include "MyBGSubtractorColor.h"

MyBGSubtractorColor::MyBGSubtractorColor() {
  cv::namedWindow(win_trackbars_);

  cv::createTrackbar("H low:", win_trackbars_, &h_low_, 100);
  cv::createTrackbar("H up:", win_trackbars_, &h_up_, 100);
  cv::createTrackbar("L low:", win_trackbars_, &l_low_, 100);
  cv::createTrackbar("L up:", win_trackbars_, &l_up_, 100);
  cv::createTrackbar("S low:", win_trackbars_, &s_low_, 100);
  cv::createTrackbar("S up:", win_trackbars_, &s_up_, 100);
  cv::createTrackbar("Temp frame number:", win_trackbars_,
                     &temp_frame_number, max_samples_ - 1);
}

MyBGSubtractorColor::~MyBGSubtractorColor() {
  cv::destroyWindow(win_trackbars_);
}

void MyBGSubtractorColor::LearnModel(cv::VideoCapture &cap) {
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

  cv::Mat frame;

  std::vector<cv::Point> samples_positions;

  while (true) {
    cap >> frame;

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

        cv::rectangle(tmp_frame, cv::Rect(p.x, p.y, sample_size_, sample_size_),
                      cv::Scalar(0, 255, 0), 2);
      }
    }

    cv::flip(tmp_frame, tmp_frame, 1);
    cv::imshow(window_samples, tmp_frame);

    if (cv::waitKey(40) == ' ')
      break;
  }

  cv::destroyWindow(window_samples);

  // CODIGO 1.1
  // Obtener las regiones de interés y calcular la media de cada una de ellas
  // almacenar las medias en la variable means_

  cv::Mat hls_frame;
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  for (const auto &sample : samples_positions) {
    cv::Mat roi =
        hls_frame(cv::Rect(sample.x, sample.y, sample_size_, sample_size_));
    means_.push_back(cv::mean(roi));
  }
}

void MyBGSubtractorColor::ObtainBGMask(cv::Mat frame, cv::Mat &bgmask) {
  cv::Mat acc = cv::Mat::zeros(frame.size(), CV_8U);

  cv::Mat hls_frame;
  cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  for (size_t i = 0; i < means_.size(); ++i) {
    cv::Scalar mean = means_[i];

    cv::Scalar low_bound(mean[0] - h_low_, mean[1] - s_low_, mean[2] - l_low_);
    cv::Scalar up_bound(mean[0] + h_up_, mean[1] + s_up_, +mean[2] + l_up_);

    clamp(low_bound, 0, 255);
    clamp(up_bound, 0, 255);

    cv::Mat temp_bgmask;
    cv::inRange(hls_frame, low_bound, up_bound, temp_bgmask);
    acc += temp_bgmask;

    // For debugging, show temp masks
    if (i == temp_frame_number) {
      cv::flip(temp_bgmask, temp_bgmask, 1);
      cv::imshow(win_trackbars_, temp_bgmask);
    }
  }

  acc.copyTo(bgmask);
}

void MyBGSubtractorColor::clamp(cv::Scalar &s, int low, int up) {
  s[0] = (s[0] < low) ? low : (s[0] > up) ? up : s[0];
  s[1] = (s[1] < low) ? low : (s[1] > up) ? up : s[1];
  s[2] = (s[2] < low) ? low : (s[2] > up) ? up : s[2];
}
