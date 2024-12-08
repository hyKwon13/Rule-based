import cv2
import numpy as np

drawing = False  # 마우스 드래그 상태를 표시하는 플래그
ix, iy = -1, -1  # 마우스로 드래그를 시작한 좌표를 저장할 변수
position_roi_coordinates = None  # ROI(관심영역) 좌표 저장용 변수
roi_selected = False  # ROI가 선택되었는지 여부를 나타내는 플래그

def create_rotated_templates(roi, num_steps=12):
    """
    주어진 ROI 이미지를 다양한 각도로 회전한 템플릿 이미지를 생성하는 함수.

    - 매칭을 수행할 때 물체의 회전으로 인한 불일치를 줄이기 위해,
      ROI를 여러 각도로 회전한 템플릿들을 사전에 준비합니다.
    - canny 에지 검출을 통해 경계선 정보만을 사용해 매칭 효율을 높입니다.
    - num_steps에 따라 -30도에서 +30도 사이를 일정한 각도로 분할하여 템플릿을 생성합니다.
    """
    # ROI를 그레이 스케일로 변환 후 에지 검출
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges_roi = cv2.Canny(gray_roi, 100, 200)
    
    h, w = edges_roi.shape[:2]
    center = (w // 2, h // 2)  # 회전의 중심점 (ROI 중앙)
    
    # 회전 각도 계산: -30도에서 +30도를 num_steps 개수로 나눔
    angle_step = 60 / (num_steps - 1)
    angles = [-30 + angle_step * i for i in range(num_steps)]
    
    rotated_templates = []
    for angle in angles:
        rotated, ang = rotate_image(edges_roi, center, angle, w, h)
        rotated_templates.append((rotated, ang))

    return rotated_templates

def rotate_image(edges_roi, center, angle, w, h):
    """
    주어진 이미지를 특정 각도(angle)로 회전한 이미지를 반환하는 함수.
    
    - cv2.getRotationMatrix2D를 사용해 회전 변환 행렬을 구한 뒤,
      warpAffine을 이용해 이미지를 실제 회전시킵니다.
    """
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(edges_roi, M, (w, h))
    return rotated, angle

def perform_position_correction(frame, position_roi_coordinates, rotated_templates):
    """
    현재 프레임에서 ROI 템플릿과의 매칭을 통해 물체의 위치를 찾아내고,
    그 위치에 회전 정보까지 반영하여 박스를 표시하는 함수.
    
    - matchTemplate를 활용해 각 회전된 템플릿을 프레임 내 에지 이미지와 비교
      가장 높은 매칭 점수를 가진 위치와 각도를 선택.
    - 일정 임계값 이상일 경우 해당 위치를 박스로 표시.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges_frame = cv2.Canny(gray_frame, 100, 200)

    best_val = -1
    best_loc = (0, 0)
    best_angle = 0

    # 미리 만들어둔 여러 각도별 템플릿으로 매칭을 수행하며 최고 매칭값 탐색
    for template, angle in rotated_templates:
        res = cv2.matchTemplate(edges_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_angle = angle

    # 최고 매칭값이 일정 수준 이상이면(0.01) 위치 보정 성공으로 판단
    if best_val > 0.01:
        top_left = best_loc
        template = rotated_templates[0][0]  # 원래 ROI크기를 알기 위해 첫 템플릿 사용
        h, w = template.shape[:2]
        
        # 매칭된 위치를 중심으로 회전된 박스 좌표 계산
        rect_center = (top_left[0] + w // 2, top_left[1] + h // 2)
        box = cv2.boxPoints(((rect_center[0], rect_center[1]), (w, h), best_angle))
        box = np.int32(box)

        # 찾은 위치에 회전 박스 그리기
        cv2.polylines(frame, [box], True, (0, 255, 0), 2)
        return frame, True
    else:
        return frame, False

def mouse_callback(event, x, y, flags, param):
    """
    마우스 콜백 함수.
    마우스 이벤트를 통해 ROI를 설정하기 위한 좌표를 받아오는 역할.
    - 마우스 왼쪽 버튼을 누른 상태에서 드래그한 영역을 ROI로 설정.
    """
    global ix, iy, drawing, position_roi_coordinates, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 드래그 시작
        drawing = True
        ix, iy = x, y
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        # 드래그 중일 때 현재 드래그 영역을 표시
        if drawing:
            frame_copy = param.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 4)
            cv2.imshow('Position Correction', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 드래그 종료 후 ROI 확정
        drawing = False
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

    # 카메라 열기 확인
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 아직 ROI를 선택하지 않은 경우, 마우스 콜백 설정 및 ROI 선택 유도
        if not roi_selected:
            cv2.imshow('Position Correction', frame)
            cv2.setMouseCallback('Position Correction', mouse_callback, param=frame)
        else:
            # ROI가 선택되었지만 템플릿을 아직 만들지 않았다면, ROI로부터 템플릿 생성
            if rotated_templates is None:
                top_left_pt, bottom_right_pt = position_roi_coordinates
                # ROI좌표가 프레임 범위 안에 들어가도록 조정
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
                
                # 선택한 ROI로부터 회전 템플릿 생성
                rotated_templates = create_rotated_templates(roi)
                print("ROI가 설정되었습니다.")

            # 현재 프레임에서 위치 보정을 수행
            frame_corrected, success = perform_position_correction(frame, position_roi_coordinates, rotated_templates)
            if success:
                cv2.putText(frame_corrected, "Rotated object detected", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame_corrected, "Rotated object detection failed", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Position Correction', frame_corrected)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
