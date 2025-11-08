🛡️ YOLO 기반 CCTV 안전 모니터링 시스템
📌 프로젝트 소개

이 프로젝트는 YOLOv8 모델을 활용하여 CCTV 영상에서 안전모(Helmet), 안전조끼(Vest), 낙상(Fall) 여부를 실시간으로 감지하고, 웹 애플리케이션(Flask)으로 시각화 및 로그 관리 기능을 제공합니다.

🚀 주요 기능

데이터 처리

devide.py: 원본 데이터를 8:2 비율로 train/val 분할

update_labels.py: 라벨 클래스 ID 일괄 수정

data.yaml: 데이터셋 정의 및 클래스 매핑

모델 학습 & 평가

train.py: YOLOv8 학습 스크립트

inference.py: 테스트셋 성능 검증 (mAP, Precision, Recall 출력)

detection_save_img.py: 무작위 이미지 추론 결과 저장

실시간 탐지

real_time_inference.py: 웹캠 실시간 객체 감지

detector.py: YOLO 감지 모듈 (Flask 백엔드에서 사용)

웹 서비스 (Flask)

사용자 인증 (회원가입 / 로그인)

CCTV 선택 화면 (cctv_select.html)

실시간 감지 & 로그 기록 (logs.html)

기본 정보 페이지 (about.html)

<div align=center> 
<img src="https://github.com/user-attachments/assets/faa0b405-7153-460a-89bc-cb544d03e3f3" width="600" height="600" ></img><br/>
<시스템 구성도>
</div>

<div align=center> 
<img src="https://github.com/user-attachments/assets/15db4668-1028-4f41-986c-888c8c06aa85" width="600" height="600" ></img><br/>
<모델을 활용한 객체 탐지 결과>
</div>


▶️ 실행 방법
1. 모델 학습
python train.py

2. 성능 평가
python inference.py

3. 무작위 추론 결과 저장
python detection_save_img.py

4. 실시간 웹캠 감지
python real_time_inference.py

5. Flask 웹 실행

CCTV 선택 → 실시간 탐지 → 로그 확인 가능

📂 파일 구조
```
.
├── Code/
│   ├── data.yaml              # 데이터셋 정의
│   ├── train.py               # YOLOv8 학습
│   ├── inference.py           # 성능 검증
│   ├── detection_save_img.py  # 무작위 이미지 추론 저장
│   ├── devide.py              # train/val 데이터 분할
│   ├── update_labels.py       # 라벨 ID 수정
│   ├── real_time_inference.py # 실시간 웹캠 감지
│   ├── detector.py            # YOLO 감지 모듈
│   ├── tree.py                # 폴더 구조 출력
│   ├── templates/             # Flask HTML 템플릿
│   │   ├── home.html
│   │   ├── signup.html
│   │   ├── cctv_select.html
│   │   ├── stream.html
│   │   ├── login.html
│   │   ├── logs.html
│   │   ├── base.html
│   │   └── about.html
│   └── static/
│       └── site.css           # 웹 스타일시트
├── .vscode/
│   ├── launch.json            # VSCode 디버깅 설정
│   └── settings.json          # Python 환경 설정
└── README.md
```

🛠️ 기술 스택

Deep Learning: YOLOv8 (Ultralytics)

Web Framework: Flask (Python)

Frontend: HTML + CSS (반응형 UI)

Database: SQLite (로그 저장용, 추후 확장 가능)

IDE/환경: VSCode (launch.json, settings.json 설정 포함)

📌 향후 개선 사항

사용자별 CCTV 권한 관리

로그 데이터베이스 시각화 대시보드 추가

모바일 환경 최적화

YOLO 모델 경량화 및 성능 개선
