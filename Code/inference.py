"""
모델의 성능을 테스트 데이터셋에서 평가하고, 주요 지표를 출력하는 코드
"""

from ultralytics import YOLO


def evaluate_model_on_test_set():
    model_path = r"C:\GIT\Capstone\Web\flask-template\weights/best.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return

    print("모델 성능 검증을 시작합니다...")
    metrics = model.val(data=r"D:\MDO\fall_hat\Code\data.yaml", split="val")

    print("성능 검증이 완료되었습니다.")
    # =======================================================================================
    print("\n" + "=" * 50)
    print("모델 성능 최종 결과")
    print("=" * 50)
    print(f"\nmAP50-95 (종합 성능 점수): {metrics.box.map:.4f}")
    print("이 점수가 모델의 전반적인 정확도를 나타내는 가장 중요한 핵심 지표입니다.")
    print(f"\nmAP50 (너그로운 기준의 정확도): {metrics.box.map50:.4f}")
    print(
        "예측 박스와 실제 박스가 50% 이상 겹치면 정답으로 인정하는 기준의 점수입니다."
    )
    print("\n" + "-" * 50)
    print("클래스별 상세 분석")
    print("-" * 50)
    # =======================================================================================

    class_names = model.names
    per_class_precision = metrics.box.p
    per_class_recall = metrics.box.r
    per_class_map50 = metrics.box.ap50

    for i, name in class_names.items():
        precision = per_class_precision[i]
        recall = per_class_recall[i]
        map50 = per_class_map50[i]
        print(f"\n클래스: '{name}'")
        print(
            f"  - 정밀도 (Precision): {precision:.4f}  (모델이 '{name}'이라고 예측한 것 중, 실제 '{name}'이었던 비율)"
        )
        print(
            f"  - 재현율 (Recall): {recall:.4f}     (실제 존재하는 모든 '{name}' 중에서, 모델이 감지해낸 비율)"
        )
        print(f"  - mAP@.50: {map50:.4f}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    evaluate_model_on_test_set()
