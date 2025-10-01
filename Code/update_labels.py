"""
라벨 파일 내부를 훑으면서 특정 class ID만 골라서 새 ID로 치환하는 코드
"""

import os


def update_class_id(target_folders, old_class_id, new_class_id):
    old_id_str = str(old_class_id)
    new_id_str = str(new_class_id)
    total_files_modified = 0
    total_lines_changed = 0

    print(
        f"클래스 ID '{old_id_str}'을(를) '{new_id_str}'(으)로 변경하는 작업을 시작합니다..."
    )

    for folder_path in target_folders:
        if not os.path.exists(folder_path):
            print(f"경고: '{folder_path}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
            continue

        print(f"\n--- '{folder_path}' 폴더 작업 중 ---")

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                updated_lines = []
                file_modified = False

                with open(file_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if parts and parts[0] == old_id_str:
                        parts[0] = new_id_str
                        updated_lines.append(" ".join(parts) + "\n")
                        file_modified = True
                        total_lines_changed += 1
                    else:
                        updated_lines.append(line)

                if file_modified:
                    with open(file_path, "w") as f:
                        f.writelines(updated_lines)
                    print(f"'{filename}' 파일의 클래스 ID를 변경했습니다.")
                    total_files_modified += 1

    print("\n" + "=" * 40)
    print("작업이 완료되었습니다.")
    print(f"총 {total_files_modified}개의 파일을 수정했습니다.")
    print(f"총 {total_lines_changed}개의 라인을 변경했습니다.")
    print("=" * 40)


if __name__ == "__main__":
    folders_to_process = [
        r"D:\MDO\fall_hat\temp\hat\labels\test",
        r"D:\MDO\fall_hat\temp\hat\labels\train",
        r"D:\MDO\fall_hat\temp\hat\labels\val",
    ]

    original_id = 0  # 바꿀 대상 ID
    target_id = 2  # 새로 지정할 ID

    update_class_id(folders_to_process, original_id, target_id)
