from datetime import datetime
import time
import cv2
import numpy as np
import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
    stream_with_context,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
    UserMixin,
)
from werkzeug.security import generate_password_hash, check_password_hash
from model.detector import get_detector, Detection

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key-change-me"

# 절대 경로로 DB 고정(작업 폴더 달라도 동일 DB 사용)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# ---------- 모델 ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(
        db.String(64), unique=True, nullable=False, index=True
    )  # 로그인 ID
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    logs = db.relationship("DetectionLog", backref="user", lazy=True)

    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)  # 예: 정문, 로비 등
    stream_url = db.Column(
        db.String(255), nullable=False
    )  # 0(웹캠), 파일경로, rtsp/http 등
    logs = db.relationship("DetectionLog", backref="camera", lazy=True)


class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey("camera.id"), nullable=False)
    label = db.Column(db.String(64), nullable=False)  # 예: person
    confidence = db.Column(db.Float, nullable=False)  # 0.0 ~ 1.0
    ts = db.Column(db.DateTime, default=datetime.utcnow, index=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------- 라우트: 인증 ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("로그인에 성공했습니다.", "success")
            next_url = request.args.get("next") or url_for("cctv_select")
            return redirect(next_url)
        flash("아이디 또는 비밀번호가 올바르지 않습니다.", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("모든 필드를 입력해 주세요.", "warning")
            return render_template("signup.html")

        if User.query.filter(
            (User.username == username) | (User.email == email)
        ).first():
            flash("이미 사용 중인 아이디 또는 이메일입니다.", "danger")
            return render_template("signup.html")

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("회원가입이 완료되었습니다. 로그인해 주세요.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("로그아웃되었습니다.", "info")
    return redirect(url_for("login"))


# ---------- 라우트: CCTV 선택/스트림 ----------
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("cctv_select"))
    return redirect(url_for("login"))


@app.route("/cctv")
@login_required
def cctv_select():
    cams = Camera.query.order_by(Camera.id.asc()).all()
    return render_template("cctv_select.html", cameras=cams)


@app.route("/cctv/<int:camera_id>")
@login_required
def cctv_page(camera_id: int):
    cam = db.session.get(Camera, camera_id)
    if not cam:
        flash("존재하지 않는 카메라입니다.", "warning")
        return redirect(url_for("cctv_select"))
    return render_template("stream.html", camera=cam)


@app.route("/cctv/<int:camera_id>/video_feed")
@login_required
def video_feed(camera_id: int):
    cam = db.session.get(Camera, camera_id)
    if not cam:
        return "Camera not found", 404

    source = 0 if cam.stream_url == "0" else cam.stream_url
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return "Cannot open video source", 500

    # current_user가 사라지기 전에 캡처
    uid = int(current_user.id)
    cid = int(cam.id)

    # 모델 싱글톤 준비(최초 호출 시 로드)
    detector = get_detector()

    last_log_ts = 0.0
    log_interval = 2.0

    def gen():
        nonlocal last_log_ts
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ----- 실제 모델 추론 -----
            detections = detector.predict(frame)

            # 시각화
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{det.label} {det.confidence:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 주기적 로그(가장 높은 신뢰도 1건)
            now = time.time()
            if detections and now - last_log_ts >= log_interval:
                top = max(detections, key=lambda d: d.confidence)
                db.session.add(
                    DetectionLog(
                        user_id=uid,
                        camera_id=cid,
                        label=top.label,
                        confidence=top.confidence,
                    )
                )
                db.session.commit()
                last_log_ts = now

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            jpg = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

        cap.release()

    return Response(
        stream_with_context(gen()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------- 라우트: 로그 조회 ----------
@app.route("/logs")
@login_required
def logs():
    camera_id = request.args.get("camera_id", type=int)
    q = DetectionLog.query.filter_by(user_id=current_user.id)
    if camera_id:
        q = q.filter_by(camera_id=camera_id)
    q = q.order_by(DetectionLog.ts.desc()).limit(200)
    cams = Camera.query.order_by(Camera.id.asc()).all()
    return render_template(
        "logs.html", logs=q.all(), cameras=cams, selected_camera_id=camera_id
    )


# ---------- CLI: DB 초기화/시드 ----------
@app.cli.command("init-db")
def init_db_command():
    """데이터베이스 테이블 생성 및 기본 카메라 1개 시드"""
    db.create_all()
    if Camera.query.count() == 0:
        db.session.add(Camera(name="정문 공사", stream_url="0"))
        db.session.commit()
        print("DB 초기화 및 기본 카메라(정문 공사, 0) 시드 완료")
    else:
        print("DB가 이미 초기화되어 있습니다.")


@app.cli.command("set-one-camera")
def set_one_camera():
    """로그/카메라 초기화 후 카메라 1개(정문 공사, 0)만 유지"""
    # 탐지 로그 먼저 삭제(외래키 충돌 방지)
    DetectionLog.query.delete()
    Camera.query.delete()
    db.session.add(Camera(name="정문 공사", stream_url="0"))
    db.session.commit()
    print("카메라를 1개(정문 공사, 0)로 재설정 완료")


@app.cli.command("seed-cameras")
def seed_cameras():
    """샘플 카메라 15개 시드"""
    names = [
        "정문 공사",
        "후문 공사",
        "자재 창고",
        "타워크레인 상부",
        "타워크레인 하부",
        "지하 1층 출입구",
        "지상 1층 로비",
        "2층 배선 구역",
        "3층 용접 구역",
        "옥상 난간",
        "주차장 램프",
        "외벽 거푸집",
        "폐기물 집하장",
        "안전 교육장",
        "현장 사무실",
    ]
    # 첫 번째는 로컬 웹캠(0), 나머지는 예시 소스(없으면 클릭 시 500 반환될 수 있음)
    urls = ["0"] + ["1"] * (len(names) - 1)
    existing = {c.name for c in Camera.query.all()}
    added = 0
    for n, u in zip(names, urls):
        if n in existing:
            continue
        db.session.add(Camera(name=n, stream_url=u))
        added += 1
    if added:
        db.session.commit()
        print(f"{added}개 카메라 시드 완료")
    else:
        print("추가할 카메라가 없습니다(이미 존재).")


if __name__ == "__main__":
    # 개발 편의를 위해 직접 실행도 가능(자동 시드 제거)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
