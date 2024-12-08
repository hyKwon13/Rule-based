import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

drawing = False  # 마우스 드래그 상태 확인
ix, iy = -1, -1  # 시작 좌표
position_roi_coordinates = None  # ROI 좌표 저장
roi_selected = False  # ROI 선택 여부

def create_rotated_templates(roi, num_steps=12):
    """주어진 ROI에 대해 회전된 템플릿을 생성합니다."""
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges_roi = cv2.Canny(gray_roi, 100, 200)
    h, w = edges_roi.shape[:2]
    center = (w // 2, h // 2)
    rotated_templates = [(edges_roi, 0)]
    angle_step = 60 / (num_steps - 1)  # 회전 각도 스텝 계산

    with ThreadPoolExecutor() as executor:
        angles = [-30 + angle_step * i for i in range(num_steps)]
        futures = [executor.submit(rotate_image, edges_roi, center, angle, w, h) for angle in angles]
        rotated_templates.extend([future.result() for future in futures])

    return rotated_templates

def rotate_image(edges_roi, center, angle, w, h):
    """이미지를 주어진 각도로 회전합니다."""
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(edges_roi, M, (w, h))
    return rotated, angle

def perform_position_correction(frame, position_roi_coordinates, rotated_templates):
    """위치 보정을 수행하고 결과를 반환합니다."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges_frame = cv2.Canny(gray_frame, 100, 200)

    best_val = -1
    best_loc = (0, 0)
    best_angle = 0

    for template, angle in rotated_templates:
        res = cv2.matchTemplate(edges_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_angle = angle

    if best_val > 0.01:
        top_left = best_loc
        template = rotated_templates[0][0]
        h, w = template.shape[:2]
        rect_center = (top_left[0] + w // 2, top_left[1] + h // 2)
        box = cv2.boxPoints(((rect_center[0], rect_center[1]), (w, h), best_angle))
        box = np.int32(box)  # 수정된 부분

        cv2.polylines(frame, [box], True, (0, 255, 0), 4)
        return frame, True
    else:
        return frame, False

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, position_roi_coordinates, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = param.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Position Correction', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 드래그 방향에 따른 좌표 정렬
        x_min = min(ix, x)
        y_min = min(iy, y)
        x_max = max(ix, x)
        y_max = max(iy, y)
        position_roi_coordinates = ((x_min, y_min), (x_max, y_max))
        roi_selected = True
        print("ROI 선택됨:", position_roi_coordinates)

def main():
    global roi_selected, position_roi_coordinates
    cap = cv2.VideoCapture(0)
    roi_selected = False
    rotated_templates = None

    # 카메라 열림 확인
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        if not roi_selected:
            cv2.imshow('Position Correction', frame)
            cv2.setMouseCallback('Position Correction', mouse_callback, param=frame)
        else:
            if rotated_templates is None:
                top_left_pt, bottom_right_pt = position_roi_coordinates
                # 좌표가 프레임 크기를 벗어나지 않도록 클램프
                x1 = max(0, top_left_pt[0])
                y1 = max(0, top_left_pt[1])
                x2 = min(frame.shape[1], bottom_right_pt[0])
                y2 = min(frame.shape[0], bottom_right_pt[1])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    print("유효한 ROI가 아닙니다. 다시 선택해주세요.")
                    roi_selected = False
                    rotated_templates = None
                    continue
                rotated_templates = create_rotated_templates(roi)
                print("ROI가 설정되었습니다.")

            frame_corrected, success = perform_position_correction(frame, position_roi_coordinates, rotated_templates)
            if success:
                cv2.putText(frame_corrected, "Position Corrected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame_corrected, "Position Correction Failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Position Correction', frame_corrected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
