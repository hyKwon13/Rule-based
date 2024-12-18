#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <future>
#include <cmath>

// 전역 변수
static bool drawing = false;
static int ix = -1, iy = -1;
static bool roi_selected = false;
static std::pair<cv::Point, cv::Point> position_roi_coordinates;

// 함수 선언
static void mouse_callback(int event, int x, int y, int flags, void* param);
cv::Mat rotate_image(const cv::Mat& edges_roi, const cv::Point& center, double angle, int w, int h);
std::vector<std::pair<cv::Mat, double>> create_rotated_templates(const cv::Mat& roi, int num_steps=12);
std::pair<cv::Mat, bool> perform_position_correction(cv::Mat frame, 
                                                     const std::pair<cv::Point, cv::Point>& position_roi_coordinates, 
                                                     const std::vector<std::pair<cv::Mat,double>>& rotated_templates);

int main() {
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    std::vector<std::pair<cv::Mat,double>> rotated_templates;

    while (true) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if(!ret) {
            std::cerr << "프레임을 읽을 수 없습니다." << std::endl;
            break;
        }

        cv::namedWindow("Position Correction", cv::WINDOW_AUTOSIZE);
        if(!roi_selected) {
            cv::imshow("Position Correction", frame);
            cv::setMouseCallback("Position Correction", mouse_callback, (void*)&frame);
        } else {
            if(rotated_templates.empty()) {
                // ROI 추출
                cv::Point top_left_pt = position_roi_coordinates.first;
                cv::Point bottom_right_pt = position_roi_coordinates.second;
                // 좌표 클램프
                int x1 = std::max(0, top_left_pt.x);
                int y1 = std::max(0, top_left_pt.y);
                int x2 = std::min(frame.cols, bottom_right_pt.x);
                int y2 = std::min(frame.rows, bottom_right_pt.y);

                if (x1 >= x2 || y1 >= y2) {
                    std::cout << "유효한 ROI가 아닙니다. 다시 선택해주세요." << std::endl;
                    roi_selected = false;
                    rotated_templates.clear();
                    continue;
                }

                cv::Mat roi = frame(cv::Rect(cv::Point(x1,y1), cv::Point(x2,y2))).clone();
                if (roi.empty()) {
                    std::cout << "유효한 ROI가 아닙니다. 다시 선택해주세요." << std::endl;
                    roi_selected = false;
                    rotated_templates.clear();
                    continue;
                }
                rotated_templates = create_rotated_templates(roi);
                std::cout << "ROI가 설정되었습니다." << std::endl;
            }

            auto result = perform_position_correction(frame, position_roi_coordinates, rotated_templates);
            cv::Mat frame_corrected = result.first;
            bool success = result.second;
            if (success) {
                cv::putText(frame_corrected, "Position Corrected", cv::Point(50,50), 
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0),2);
            } else {
                cv::putText(frame_corrected, "Position Correction Failed", cv::Point(50,50), 
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255),2);
            }

            cv::imshow("Position Correction", frame_corrected);
        }

        if((cv::waitKey(1) & 0xFF) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

static void mouse_callback(int event, int x, int y, int flags, void* param) {
    cv::Mat* frame = static_cast<cv::Mat*>(param);

    switch(event) {
        case cv::EVENT_LBUTTONDOWN:
            drawing = true;
            ix = x;
            iy = y;
            roi_selected = false;
            break;
        case cv::EVENT_MOUSEMOVE:
            if(drawing && frame) {
                cv::Mat frame_copy = frame->clone();
                cv::rectangle(frame_copy, cv::Point(ix, iy), cv::Point(x, y), cv::Scalar(0,255,0),2);
                cv::imshow("Position Correction", frame_copy);
            }
            break;
        case cv::EVENT_LBUTTONUP:
            drawing = false;
            {
                int x_min = std::min(ix, x);
                int y_min = std::min(iy, y);
                int x_max = std::max(ix, x);
                int y_max = std::max(iy, y);
                position_roi_coordinates = std::make_pair(cv::Point(x_min, y_min), cv::Point(x_max, y_max));
                roi_selected = true;
                std::cout << "ROI 선택됨: ((" << x_min << "," << y_min << "),(" << x_max << "," << y_max << "))" << std::endl;
            }
            break;
    }
}

cv::Mat rotate_image(const cv::Mat& edges_roi, const cv::Point& center, double angle, int w, int h) {
    cv::Mat M = cv::getRotationMatrix2D(center, -angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(edges_roi, rotated, M, cv::Size(w,h));
    return rotated;
}

std::vector<std::pair<cv::Mat, double>> create_rotated_templates(const cv::Mat& roi, int num_steps) {
    cv::Mat gray_roi;
    cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);

    cv::Mat edges_roi;
    cv::Canny(gray_roi, edges_roi, 100, 200);

    int h = edges_roi.rows;
    int w = edges_roi.cols;
    cv::Point center(w/2, h/2);

    std::vector<std::pair<cv::Mat,double>> rotated_templates;
    rotated_templates.push_back(std::make_pair(edges_roi, 0.0));

    double angle_step = 60.0 / (num_steps - 1);
    std::vector<std::future<std::pair<cv::Mat,double>>> futures;
    for (int i=0; i<num_steps; i++) {
        double angle = -30.0 + angle_step * i;
        futures.push_back(std::async(std::launch::async, [=]() {
            cv::Mat r = rotate_image(edges_roi, center, angle, w, h);
            return std::make_pair(r, angle);
        }));
    }

    for (auto &f : futures) {
        rotated_templates.push_back(f.get());
    }
    return rotated_templates;
}

std::pair<cv::Mat, bool> perform_position_correction(cv::Mat frame, 
                                                     const std::pair<cv::Point, cv::Point>& position_roi_coordinates, 
                                                     const std::vector<std::pair<cv::Mat,double>>& rotated_templates) {
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::Mat edges_frame;
    cv::Canny(gray_frame, edges_frame, 100,200);

    double best_val = -1;
    cv::Point best_loc(0,0);
    double best_angle = 0.0;

    for (auto &tpl : rotated_templates) {
        const cv::Mat& template_img = tpl.first;
        double angle = tpl.second;
        cv::Mat res;
        cv::matchTemplate(edges_frame, template_img, res, cv::TM_CCOEFF_NORMED);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);
        if (maxVal > best_val) {
            best_val = maxVal;
            best_loc = maxLoc;
            best_angle = angle;
        }
    }

    if (best_val > 0.01) {
        const cv::Mat& template_ref = rotated_templates[0].first;
        int th = template_ref.rows;
        int tw = template_ref.cols;

        cv::Point rect_center(best_loc.x + tw/2, best_loc.y + th/2);

        // RotatedRect 생성 후 box 포인트 계산
        cv::RotatedRect rRect(cv::Point2f((float)rect_center.x, (float)rect_center.y),
                              cv::Size2f((float)tw, (float)th),
                              (float)best_angle);

        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i=0; i<4; i++) {
            // vertices[i]는 실수 좌표이므로 int로 변환
            vertices[i] = cv::Point2f((float)((int)vertices[i].x), (float)((int)vertices[i].y));
        }

        for (int i=0; i<4; i++)
            cv::line(frame, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0),4);

        return std::make_pair(frame, true);
    } else {
        return std::make_pair(frame, false);
    }
}
