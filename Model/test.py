import torch
from torchvision import models, transforms
from PIL import Image

# 데이터 전처리 (사용자 이미지에 맞는 전처리)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 클래스 이름 (학습 시 사용한 클래스와 동일해야 함)
class_names = ['Belly', 'Ear', 'Elbow', 'Eye', 'Foot', 'Hand', 'Knee', 'Neck', 'Nose', 'Shoulders']  # 예시로 클래스 이름을 직접 넣습니다.

# 모델 불러오기
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 구조 설정 (학습 시 사용한 것과 동일)
model = models.resnet50(weights="IMAGENET1K_V1")  # pretrained=True 대신 weights 사용
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # class_names의 크기만큼 출력
model = model.to(device)

# 학습된 모델 가중치 불러오기
model.load_state_dict(torch.load("D:/Capstone/body_parts_resnet50.pth"))
model.eval()  # 평가 모드로 설정

# 이미지 분류 함수
def classify_image(image_path):
    # 이미지 열기
    image = Image.open(image_path)

    # 이미지 전처리
    image = data_transforms(image).unsqueeze(0)  # 배치 차원 추가

    # 장치에 올리기 (CPU 또는 GPU)
    image = image.to(device)

    # 모델 예측
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    # 예측된 클래스 라벨
    predicted_class = class_names[preds.item()]
    
    return predicted_class

# 사용자 이미지 경로 지정 (예시)
image_path = "C:/Users/as/Desktop/HHAANNDD.jpg"  # 여기에 실제 이미지 파일 경로를 입력하세요

# 이미지 분류 실행
predicted_class = classify_image(image_path)
print(f"Predicted class: {predicted_class}")
