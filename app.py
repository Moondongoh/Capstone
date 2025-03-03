from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

app = Flask(__name__)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 클래스 이름 (모델 학습 시 사용한 순서대로)
class_names = ['Belly', 'Ear', 'Elbow', 'Eye', 'Foot', 'Hand', 'Knee', 'Neck', 'Nose', 'Shoulders']

# 모델 로드
class ResNetModel(torch.nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(weights=None)  # 최신 방식
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_names))  # 클래스 개수 설정
        model_path = r"C:\Github\Capstone\body_parts_resnet50.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, img):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(img)
            probs = torch.nn.functional.softmax(output, dim=1)  # 확률 변환
            confidence, predicted = torch.max(probs, 1)  # 최고 확률 및 클래스
        return class_names[predicted.item()], confidence.item()  # 클래스 이름과 확률 반환

# 모델 인스턴스 생성
model = ResNetModel()

# 로그인 페이지
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form["id"]
        password = request.form["password"]
        if user_id == "firefighter" and password == "1234":
            return redirect(url_for("select_member"))
        else:
            return render_template("login.html", error="Invalid ID or Password")
    return render_template("login.html")

# 대원 선택 페이지
@app.route("/select_member", methods=["GET", "POST"])
def select_member():
    members = ["대원 1", "대원 2", "대원 3", "대원 4"]
    return render_template("select_member.html", members=members)

# 웹캠 스트리밍 함수 (바운딩 박스 + 확률 0.9 이상 필터링)
def generate_frames():
    camera = cv2.VideoCapture(0)  # 0번 카메라 사용
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        # OpenCV 이미지를 PIL 이미지로 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        # 모델로 예측 수행
        prediction, confidence = model.predict(img_pil)

        # 확률이 0.9 이상일 경우에만 표시
        if confidence >= 0.9:
            label = f"{prediction} ({confidence:.2f})"

            # 바운딩 박스 좌표 (이미지 중앙)
            h, w, _ = frame.shape
            x1, y1 = int(w * 0.3), int(h * 0.3)
            x2, y2 = int(w * 0.7), int(h * 0.7)

            # 바운딩 박스 & 텍스트 추가
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 프레임 인코딩 후 전송
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# "대원 1" 클릭 시 웹캠 페이지로 이동
@app.route("/view_member")
def view_member():
    return render_template("view_member.html")

# 웹캠 스트리밍 라우트
@app.route("/stream")
def stream():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
