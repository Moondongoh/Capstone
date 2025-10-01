"""
원본 이미지·라벨을 8:2 비율로 train/val 디렉토리로 분할 복사하는 코드
"""

import os
import random
import shutil

# 원본 이미지 및 라벨 디렉토리
source_images_dir = r"D:\MDO\fall_hat\fall_non\Not Fall\images"
source_labels_dir = r"D:\MDO\fall_hat\fall_non\Not Fall\labels"

# 생성할 train, val 디렉토리
train_dir = r"D:\MDO\fall_hat\Dataset\train"
val_dir = r"D:\MDO\fall_hat\Dataset\val"

# 분할 비율 (train: 80%, val: 20%)
split_ratio = 0.8

# train/images, train/labels, val/images, val/labels 디렉토리 생성
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [
    f
    for f in os.listdir(source_images_dir)
    if os.path.isfile(os.path.join(source_images_dir, f))
]
random.shuffle(image_files)

# 분할 지점 계산
split_point = int(len(image_files) * split_ratio)

# train 세트와 val 세트로 나누기
train_files = image_files[:split_point]
val_files = image_files[split_point:]

# train 파일 복사
for image_file in train_files:
    # 이미지 파일 복사
    shutil.copy(
        os.path.join(source_images_dir, image_file),
        os.path.join(train_dir, "images", image_file),
    )

    # 해당 라벨 파일 복사
    label_file = (
        os.path.splitext(image_file)[0] + ".txt"
    )  # 라벨 파일 확장자가 .txt라고 가정
    shutil.copy(
        os.path.join(source_labels_dir, label_file),
        os.path.join(train_dir, "labels", label_file),
    )

# val 파일 복사
for image_file in val_files:
    # 이미지 파일 복사
    shutil.copy(
        os.path.join(source_images_dir, image_file),
        os.path.join(val_dir, "images", image_file),
    )

    # 해당 라벨 파일 복사
    label_file = (
        os.path.splitext(image_file)[0] + ".txt"
    )  # 라벨 파일 확장자가 .txt라고 가정
    shutil.copy(
        os.path.join(source_labels_dir, label_file),
        os.path.join(val_dir, "labels", label_file),
    )

print("데이터 분할이 완료되었습니다.")
print(f"Train 세트: {len(train_files)}개 파일")
print(f"Validation 세트: {len(val_files)}개 파일")
