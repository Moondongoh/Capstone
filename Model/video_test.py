#확률 계산 방식(softmax 후 probabilities[0][preds.item()].item() 사용)
#분류 결과 반환((class_names[preds.item()], confidence))
#확률이 0.8 미만일 때(여전히 클래스 예측을 수행하나 화면에 표시 안 함)
#코드 간결성(if confidence >= 0.8:로 직접 필터링)

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 데이터 전처리 설정
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 클래스 이름 (예제)
class_names = ['Belly', 'Ear', 'Elbow', 'Eye', 'Foot', 'Hand', 'Knee', 'Neck', 'Nose', 'Shoulders']

# 모델 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# 학습된 모델 가중치 불러오기
model.load_state_dict(torch.load("C:/Users/mdh38/Desktop/body_parts_resnet50.pth", map_location=device))
model.eval()

# 이미지 분류 함수
def classify_image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][preds.item()].item()

    return class_names[preds.item()], confidence

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 분류 실행
    predicted_class, confidence = classify_image(frame)

    # 정확도가 0.8 이상일 때만 화면에 표시
    if confidence >= 0.8:
        cv2.putText(frame, f"Class: {predicted_class} ({confidence:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Classification", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()