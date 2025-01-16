#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <future>
#include <algorithm>

// ----- 전역 변수 및 설정 -----
static bool drawing = false;
static int ix = -1, iy = -1;                     // 시작 좌표
static cv::Rect positionRoiCoordinates;          // ROI 좌표를 Rect로 관리
static bool roiSelected = false;                 // ROI 선택 여부

// ----- 함수 선언부 -----
std::pair<cv::Mat, float> rotateImage(const cv::Mat& edgesRoi, const cv::Point2f& center, float angle, int w, int h);
std::vector<std::pair<cv::Mat, float>> createRotatedTemplates(const cv::Mat& roi, int numSteps = 12);
std::pair<cv::Mat, bool> performPositionCorrection(cv::Mat& frame, const cv::Rect& roiRect,
                                                   const std::vector<std::pair<cv::Mat, float>>& rotatedTemplates);
void mouseCallback(int event, int x, int y, int flags, void* userdata);

// ----- 이미지 회전 함수 -----
std::pair<cv::Mat, float> rotateImage(const cv::Mat& edgesRoi, const cv::Point2f& center, float angle, int w, int h)
{
    // 회전 행렬 계산
    cv::Mat M = cv::getRotationMatrix2D(center, -angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(edgesRoi, rotated, M, cv::Size(w, h));
    return {rotated, angle};
}

// ----- ROI에 대해 회전된 템플릿을 생성하는 함수 -----
std::vector<std::pair<cv::Mat, float>> createRotatedTemplates(const cv::Mat& roi, int numSteps)
{
    // ROI를 그레이 스케일로 변환 후 에지 추출
    cv::Mat grayRoi, edgesRoi;
    cv::cvtColor(roi, grayRoi, cv::COLOR_BGR2GRAY);
    cv::Canny(grayRoi, edgesRoi, 100, 200);

    int w = edgesRoi.cols;
    int h = edgesRoi.rows;
    cv::Point2f center(static_cast<float>(w) / 2.0f, static_cast<float>(h) / 2.0f);

    // 기본 템플릿(회전 안 된 상태)을 먼저 저장
    std::vector<std::pair<cv::Mat, float>> rotatedTemplates;
    rotatedTemplates.emplace_back(edgesRoi, 0.0f);

    // -30도 ~ +30도 범위를 (numSteps) 개로 분할하여 회전
    float angleStep = 60.0f / static_cast<float>(numSteps - 1);
    float startAngle = -30.0f;

    // 멀티스레드를 위한 future 백터
    std::vector<std::future<std::pair<cv::Mat, float>>> futures;
    futures.reserve(numSteps);

    for (int i = 0; i < numSteps; i++)
    {
        float angle = startAngle + angleStep * static_cast<float>(i);
        // 비동기 호출(std::launch::async)
        futures.push_back(std::async(std::launch::async, rotateImage, edgesRoi, center, angle, w, h));
    }

    // 결과 취합
    for (auto& fut : futures)
    {
        rotatedTemplates.push_back(fut.get());
    }

    return rotatedTemplates;
}

// ----- 객체 탐지를 수행하고 결과를 반환하는 함수 -----
std::pair<cv::Mat, bool> performPositionCorrection(cv::Mat& frame, const cv::Rect& roiRect,
                                                   const std::vector<std::pair<cv::Mat, float>>& rotatedTemplates)
{
    // 프레임을 그레이 스케일로 변환 후 에지 추출
    cv::Mat grayFrame, edgesFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::Canny(grayFrame, edgesFrame, 100, 200);

    double bestVal = -12.0;
    cv::Point bestLoc(0, 0);
    float bestAngle = 0.0f;

    // 모든 회전 템플릿을 순회하며 matchTemplate
    for (const auto& templateAndAngle : rotatedTemplates)
    {
        const cv::Mat& tmpl = templateAndAngle.first;
        float angle = templateAndAngle.second;

        cv::Mat result;
        cv::matchTemplate(edgesFrame, tmpl, result, cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        if (maxVal > bestVal)
        {
            bestVal = maxVal;
            bestLoc = maxLoc;
            bestAngle = angle;
        }
    }

    bool found = false;
    // 임계값(예: 0.01)보다 크면 검출 성공으로 판단
    if (bestVal > 0.01)
    {
        found = true;
        // 회전된 템플릿 중 0번(기본 템플릿)의 크기로 바운딩
        // 실제로는 bestAngle에 따라 달라질 수 있지만, 여기서는 단순화
        cv::Mat tmpl = rotatedTemplates[0].first;
        int w = tmpl.cols;
        int h = tmpl.rows;

        // 중심 좌표
        cv::Point2f rectCenter(bestLoc.x + static_cast<float>(w) / 2.0f,
                               bestLoc.y + static_cast<float>(h) / 2.0f);

        // OpenCV C++에서 회전 사각형 표현: RotatedRect
        cv::RotatedRect rRect(rectCenter, cv::Size2f(static_cast<float>(w), static_cast<float>(h)), bestAngle);

        // 4개의 꼭짓점 좌표 구하기
        cv::Point2f vertices[4];
        rRect.points(vertices);

        // 정수 좌표로 변환 후 폴리라인 그리기
        std::vector<cv::Point> points;
        for (int i = 0; i < 4; i++)
        {
            points.emplace_back(vertices[i]);
        }
        cv::polylines(frame, points, true, cv::Scalar(0, 255, 0), 4);
    }

    return {frame, found};
}

// ----- 마우스 콜백 함수 -----
void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    // userdata로 현재 프레임을 받아온다.
    cv::Mat* framePtr = reinterpret_cast<cv::Mat*>(userdata);
    if (!framePtr) return;

    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        drawing = true;
        ix = x; 
        iy = y; 
        roiSelected = false;
        break;

    case cv::EVENT_MOUSEMOVE:
        if (drawing)
        {
            cv::Mat frameCopy = framePtr->clone();
            cv::rectangle(frameCopy, cv::Point(ix, iy), cv::Point(x, y), cv::Scalar(0, 255, 0), 2);
            cv::imshow("Rotated object detection", frameCopy);
        }
        break;

    case cv::EVENT_LBUTTONUP:
        drawing = false;
        {
            int x_min = std::min(ix, x);
            int y_min = std::min(iy, y);
            int x_max = std::max(ix, x);
            int y_max = std::max(iy, y);

            // Rect로 ROI 설정
            positionRoiCoordinates = cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max));
            roiSelected = true;

            std::cout << "ROI 선택됨: (" << x_min << ", " << y_min << ") ~ ("
                      << x_max << ", " << y_max << ")" << std::endl;
        }
        break;
    }
}

// ----- main 함수 -----
int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    std::vector<std::pair<cv::Mat, float>> rotatedTemplates;
    bool templatesCreated = false;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "프레임을 읽을 수 없습니다." << std::endl;
            break;
        }

        // 아직 ROI를 선택하지 않았다면, 원본 프레임을 보여주면서 마우스 콜백 설정
        if (!roiSelected)
        {
            cv::imshow("Rotated object detection", frame);
            // 마우스 콜백은 한 번만 등록해도 계속 유효
            cv::setMouseCallback("Rotated object detection", mouseCallback, &frame);
        }
        else
        {
            // ROI가 선택되었다면, 한 번만 템플릿 생성
            if (!templatesCreated)
            {
                // ROI 좌표가 유효한지 체크
                cv::Rect clampedRect = positionRoiCoordinates & cv::Rect(0, 0, frame.cols, frame.rows);
                if (clampedRect.width <= 0 || clampedRect.height <= 0)
                {
                    std::cout << "유효한 ROI가 아닙니다. 다시 선택해주세요." << std::endl;
                    roiSelected = false;
                    templatesCreated = false;
                    continue;
                }

                cv::Mat roi = frame(clampedRect).clone();
                if (roi.empty())
                {
                    std::cout << "유효한 ROI가 아닙니다. 다시 선택해주세요." << std::endl;
                    roiSelected = false;
                    templatesCreated = false;
                    continue;
                }

                rotatedTemplates = createRotatedTemplates(roi, 12);
                templatesCreated = true;
                std::cout << "ROI가 설정되었습니다." << std::endl;
            }

            // 객체 탐지 수행
            auto [frameCorrected, success] = performPositionCorrection(frame, positionRoiCoordinates, rotatedTemplates);
            if (success)
            {
                cv::putText(frameCorrected, "Rotated object detected", cv::Point(50, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
            else
            {
                cv::putText(frameCorrected, "Rotated object detection failed", cv::Point(50, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            }

            cv::imshow("Rotated object detection", frameCorrected);
        }

        // 'q' 키를 누르면 종료
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
