#include "MyBGSubtractorColor.h"

MyBGSubtractorColor::MyBGSubtractorColor() {
  cv::namedWindow(win_trackbars_, cv::WINDOW_GUI_EXPANDED);
  cv::moveWindow(win_trackbars_, 1400, 50);
  cv::namedWindow("test");
  cv::namedWindow("bg");

  bg_subtractor_ = cv::createBackgroundSubtractorMOG2(500, 5, false);
  face_subtractor_.load("res/haarcascade_frontalface_alt.xml");

  cv::createTrackbar("H low:", win_trackbars_, &h_low_, 100);
  cv::createTrackbar("H up:", win_trackbars_, &h_up_, 100);
  cv::createTrackbar("L low:", win_trackbars_, &l_low_, 100);
  cv::createTrackbar("L up:", win_trackbars_, &l_up_, 100);
  cv::createTrackbar("S low:", win_trackbars_, &s_low_, 100);
  cv::createTrackbar("S up:", win_trackbars_, &s_up_, 100);
}

MyBGSubtractorColor::~MyBGSubtractorColor() {
  cv::destroyWindow(win_trackbars_);
  cv::destroyWindow("test");
  cv::destroyWindow("bg");
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

  std::vector<cv::Point> samples_positions;

  cv::Mat frame;
  while (true) {
    cap >> frame;

    if (frame.empty()) {
      std::cerr << "Read empty frame." << std::endl;
      continue;
    }

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

  means_.clear();
  for (const auto &sample : samples_positions) {
    cv::Mat roi =
        hls_frame(cv::Rect(sample.x, sample.y, sample_size_, sample_size_));
    means_.push_back(cv::mean(roi));
    std::cout << cv::mean(roi) << std::endl;
  }
}

void MyBGSubtractorColor::LearnModel(std::istream &means_file) {
  while (means_file) {
    cv::Scalar mean;
    means_file >> mean[0] >> mean[1] >> mean[2] >> mean[3];

    if (means_file.eof())
      break;
    means_.push_back(mean);
  }

  std::cout << means_.size() << std::endl;
}

void MyBGSubtractorColor::LearnBGModel(cv::VideoCapture &cap) {
  cv::Mat frame;
  cv::Mat temp;

  while (frame.empty())
    cap >> frame;

  // Borrar el último fondo guardado
  cv::Mat hls_frame;
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);
  bg_subtractor_->apply(hls_frame, temp, 1);

  // Generar un fondo nuevo utilizando tantas imágenes como bg_samples
  for (int i = 0; i < bg_samples_; ++i) {
    cap >> frame;
    if (frame.empty())
      continue;

    cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);
    bg_subtractor_->apply(hls_frame, temp);
  }

  bg_subtractor_->getBackgroundImage(temp);
  cv::imshow("bg", temp);
}

void MyBGSubtractorColor::ObtainBGMask(cv::Mat frame, cv::Mat &bgmask) const {
  cv::Mat hls_frame;
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  if (bg_subtractor_enabled_)
    RemoveBG(hls_frame, hls_frame);

  cv::Mat acc = cv::Mat::zeros(frame.size(), CV_8U);

  for (const auto &mean : means_) {
    cv::Scalar low_bound(mean[0] - h_low_, mean[1] - s_low_, mean[2] - l_low_);
    cv::Scalar up_bound(mean[0] + h_up_, mean[1] + s_up_, +mean[2] + l_up_);

    clamp(low_bound, 0, 255);
    clamp(up_bound, 0, 255);

    cv::Mat temp_bgmask;
    cv::inRange(hls_frame, low_bound, up_bound, temp_bgmask);
    acc += temp_bgmask;
  }

  if(face_subtractor_enabled_)
    RemoveFace(frame, acc);

  acc.copyTo(bgmask);
}


void MyBGSubtractorColor::RemoveBG(cv::Mat frame, cv::Mat &masked_frame) const {
  // Get foreground mask
  cv::Mat foreground_mask;
  bg_subtractor_->apply(frame, foreground_mask, 0);
  cv::medianBlur(foreground_mask, foreground_mask, 3);

  // Get a masked frame (without most of the bg)
  cv::Mat result;
  cv::bitwise_and(frame, frame, result, foreground_mask);
  masked_frame = result;

  cv::imshow("test", masked_frame);
}


void MyBGSubtractorColor::RemoveFace(cv::Mat frame, cv::Mat &bgmask) const {
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_subtractor_.detectMultiScale(frame_gray, faces);

    for(const auto & face : faces) {
      cv::rectangle(bgmask, face, cv::Scalar(0, 0, 0), cv::FILLED);
    }
}

void MyBGSubtractorColor::clamp(cv::Scalar &s, int low, int up) const {
  s[0] = (s[0] < low) ? low : (s[0] > up) ? up : s[0];
  s[1] = (s[1] < low) ? low : (s[1] > up) ? up : s[1];
  s[2] = (s[2] < low) ? low : (s[2] > up) ? up : s[2];
}
