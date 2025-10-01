"""
YOLO 모델을 이용해 data.yaml에서 지정한 이미지(split) 중 무작위로 추출하여 추론 결과를 시각화·저장하는 코드
생성된 모델을 테스트 데이터를 이용한 추론 후 탐지 결과 저장하는 코드
"""

import os
import random
import glob
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_from_dir(d):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(d, "**", f"*{ext}"), recursive=True))
    return paths


def _collect_from_textfile(txt):
    if not os.path.isfile(txt):
        raise FileNotFoundError(f"목록 파일을 찾을 수 없음: {txt}")
    with open(txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _collect_from_any(entry):
    if isinstance(entry, str):
        if os.path.isdir(entry):
            return _collect_from_dir(entry)
        elif os.path.isfile(entry):
            _, ext = os.path.splitext(entry.lower())
            if ext == ".txt":
                return _collect_from_textfile(entry)
            elif ext in IMG_EXTS:
                return [entry]
            else:
                matched = glob.glob(entry, recursive=True)
                return [
                    p for p in matched if os.path.splitext(p)[1].lower() in IMG_EXTS
                ]
        else:
            matched = glob.glob(entry, recursive=True)
            return [p for p in matched if os.path.splitext(p)[1].lower() in IMG_EXTS]
    elif isinstance(entry, list):
        acc = []
        for e in entry:
            acc.extend(_collect_from_any(e))
        return acc
    else:
        raise ValueError(f"지원하지 않는 항목 타입: {type(entry)}")


def read_split_paths_from_yaml(yaml_path, split="val"):
    """data.yaml의 split(train/val/test) 항목을 읽어 이미지 경로 리스트 반환"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entry = data.get(split)
    if entry is None:
        raise ValueError(f"data.yaml에 '{split}' 항목이 없습니다.")

    paths = sorted(set(_collect_from_any(entry)))
    if not paths:
        raise ValueError(f"'{split}'에서 탐색된 이미지가 없습니다.")
    return paths


def save_random_predictions(
    model_path,
    data_yaml_path,
    split="test",
    k=30,
    out_project=r"D:\MDO\fall_hat\viz_samples",
    run_name="random30",
    seed=42,
    conf=0.1,
    imgsz=640,
    line_width=2,
    show_labels=True,
    show_conf=True,
    agnostic_nms=False,
):
    print("모델 로딩 중...")
    model = YOLO(model_path)

    print(f"data.yaml에서 '{split}' 이미지 경로 수집 중...")
    all_imgs = read_split_paths_from_yaml(data_yaml_path, split=split)
    print(f"'{split}' 이미지 총 {len(all_imgs)}장 발견")

    if len(all_imgs) <= k:
        sample_imgs = all_imgs
        print(f"이미지가 {k}장 이하라 전체 {len(sample_imgs)}장을 사용합니다.")
    else:
        random.seed(seed)
        sample_imgs = random.sample(all_imgs, k)
        print(f"무작위 {k}장 추출 완료.")

    print("추론 및 시각화 저장 시작...")
    results = model.predict(
        source=sample_imgs,
        save=True,
        project=out_project,
        name=run_name,
        exist_ok=True,
        conf=conf,
        imgsz=imgsz,
        line_width=line_width,
        show_labels=show_labels,
        show_conf=show_conf,
        verbose=False,
        max_det=300,
        agnostic_nms=agnostic_nms,
    )

    out_dir = os.path.join(out_project, run_name)
    print(f"완료! 시각화된 이미지가 다음 폴더에 저장되었습니다:\n  {out_dir}")
    print(f"저장 개수: {len(sample_imgs)}장")


if __name__ == "__main__":
    MODEL_PATH = (
        r"D:\MDO\fall_hat\safety_model_results\new_dataset_exp01\weights\best.pt"
    )
    DATA_YAML = r"D:\MDO\fall_hat\Code\data.yaml"

    save_random_predictions(
        model_path=MODEL_PATH,
        data_yaml_path=DATA_YAML,
        split="val",
        k=100,
        out_project=r"D:\MDO\fall_hat\viz_samples",
        run_name="test1_random30",
        seed=0,
        conf=0.1,
        imgsz=640,
        line_width=2,
        show_labels=True,
        show_conf=True,
        agnostic_nms=False,
    )
