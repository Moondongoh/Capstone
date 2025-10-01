"""
실시간 웹캠 영상을 통해 YOLOv8 모델로 객체 감지 수행하는 코드
"""

import cv2
from ultralytics import YOLO

model = YOLO(r"D:\MDO\fall_hat\safety_model_results\new_dataset_exp01\weights\best.pt")

# 0번 --> 기본 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

while True:
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Custom YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
