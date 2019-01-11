#include "MyBGSubtractorColor.h"

MyBGSubtractorColor::MyBGSubtractorColor() {
  cv::namedWindow(w_trackbars_);
  cv::moveWindow(w_trackbars_, 1400, 50);
  cv::namedWindow(w_bg_);
  cv::namedWindow(w_bg_sub_);

  bg_sub_ = cv::createBackgroundSubtractorMOG2(500, 5, false);

  cv::createTrackbar("H low:", w_trackbars_, &h_low_, 100);
  cv::createTrackbar("H up:", w_trackbars_, &h_up_, 100);
  cv::createTrackbar("L low:", w_trackbars_, &l_low_, 100);
  cv::createTrackbar("L up:", w_trackbars_, &l_up_, 100);
  cv::createTrackbar("S low:", w_trackbars_, &s_low_, 100);
  cv::createTrackbar("S up:", w_trackbars_, &s_up_, 100);
}

MyBGSubtractorColor::~MyBGSubtractorColor() {
  cv::destroyWindow(w_trackbars_);
  cv::destroyWindow(w_bg_);
  cv::destroyWindow(w_bg_sub_);
}

void MyBGSubtractorColor::LearnModel(cv::VideoCapture &cap) {
  // Crear ventana
  const std::string w_samples =
      "Cubre los cuadrados con la mano y pulsa espacio";

  cv::namedWindow(w_samples);
  cv::createTrackbar("Sample Size:", w_samples, &sample_size_, 100);
  cv::createTrackbar("Dist between Samples", w_samples,
                     &distance_between_samples_, 100);
  cv::createTrackbar("Max horiz samples", w_samples, &max_horiz_samples_, 10);
  cv::createTrackbar("Max vert samples", w_samples, &max_vert_samples_, 10);

  std::vector<cv::Point> sample_rect_positions;

  cv::Mat frame;
  while (true) {
    cap >> frame;

    if (frame.empty()) {
      std::cerr << "Read empty frame." << std::endl;
      continue;
    }

    cv::Mat tmp_frame;
    frame.copyTo(tmp_frame);

    // Mostrar los rectángulos para sacar las ROI de la piel
    sample_rect_positions.clear();
    for (int i = 0; i < max_horiz_samples_; ++i) {
      for (int j = 0; j < max_vert_samples_; ++j) {
        cv::Point p;
        p.x = frame.cols / 2 + (-max_horiz_samples_ / 2 + i) *
                                   (sample_size_ + distance_between_samples_);
        p.y = frame.rows / 2 + (-max_vert_samples_ / 2 + j) *
                                   (sample_size_ + distance_between_samples_);
        sample_rect_positions.push_back(p);

        cv::rectangle(tmp_frame, cv::Rect{p.x, p.y, sample_size_, sample_size_},
                      {0, 255, 0}, 2);
      }
    }

    cv::flip(tmp_frame, tmp_frame, 1);
    cv::imshow(w_samples, tmp_frame);

    if (cv::waitKey(40) == ' ')
      break;
  }

  cv::destroyWindow(w_samples);

  // CODIGO 1.1
  // Obtener las regiones de interés y calcular la media de cada una de ellas
  // almacenar las medias en la variable means_

  cv::Mat hls_frame;
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  means_.clear();
  for (const auto &sample : sample_rect_positions) {
    cv::Mat roi =
        hls_frame(cv::Rect{sample.x, sample.y, sample_size_, sample_size_});
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

void MyBGSubtractorColor::LearnBGModel(cv::VideoCapture &cap) const {
  cv::Mat frame;
  cv::Mat temp;

  while (frame.empty())
    cap >> frame;

  // Borrar el último fondo guardado
  cv::Mat hls_frame;
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);
  bg_sub_->apply(hls_frame, temp, 1);

  // Generar un fondo nuevo mezclando tantas imágenes como bg_samples
  for (int i = 0; i < bg_samples_; ++i) {
    cap >> frame;
    if (frame.empty())
      continue;

    cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);
    bg_sub_->apply(hls_frame, temp);
  }

  bg_sub_->getBackgroundImage(temp);
  cv::imshow(w_bg_, temp);
}

void MyBGSubtractorColor::ObtainBGMask(const cv::Mat frame,
                                       cv::Mat &bgmask) const {
  cv::Mat hls_frame;
  cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

  if (bg_sub_enabled_)
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

  acc.copyTo(bgmask);
}

void MyBGSubtractorColor::RemoveBG(const cv::Mat frame,
                                   cv::Mat &masked_frame) const {
  // Get foreground mask
  cv::Mat foreground_mask;
  bg_sub_->apply(frame, foreground_mask, 0);
  cv::medianBlur(foreground_mask, foreground_mask, 3);

  // Get a masked frame (without most of the bg)
  cv::Mat result;
  cv::bitwise_and(frame, frame, result, foreground_mask);
  masked_frame = result;

  cv::imshow(w_bg_sub_, masked_frame);
}

void MyBGSubtractorColor::clamp(cv::Scalar &s, int low, int up) const {
  s[0] = (s[0] < low) ? low : (s[0] > up) ? up : s[0];
  s[1] = (s[1] < low) ? low : (s[1] > up) ? up : s[1];
  s[2] = (s[2] < low) ? low : (s[2] > up) ? up : s[2];
}
