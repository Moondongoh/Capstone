"""
YOLOv8 모델을 새로운 데이터셋(data.yaml)으로 학습시키고 결과를 저장하는 코드
"""

from ultralytics import YOLO


def train_safety_model():
    model = YOLO("yolov8n.pt")
    try:
        print("YOLOv8 모델 학습을 시작합니다...")
        results = model.train(
            data=r"D:\MDO\fall_hat\Code/data.yaml",
            epochs=100,
            imgsz=640,
            project="safety_model_results",
            name="new_dataset_exp01",
        )
        print("모델 학습이 성공적으로 완료되었습니다.")
        print(
            f"학습된 모델은 '{results.save_dir}/weights/best.pt' 경로에 저장되었습니다."
        )

    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    train_safety_model()
