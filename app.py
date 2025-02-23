from flask import Flask, render_template, request, redirect, url_for, Response
import cv2

app = Flask(__name__)

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

# 웹캠 스트리밍 함수(대원 1을 선택할 시 웹캠 스트리밍 페이지로 이동)
def generate_frames():
    camera = cv2.VideoCapture(0)  # 0번 카메라 사용
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
