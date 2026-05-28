from pathlib import Path
import random

# 실험용 데이터셋 경로
image_folder = Path("dataset_balanced/images/train")
label_folder = Path("dataset_balanced/labels/train")

# SCRATCH 클래스 번호
scratch_class = 5

# SCRATCH 이미지를 몇 장 남길지 설정
keep_count = 800

# SCRATCH만 들어 있는 label 파일을 저장할 리스트
scratch_only_files = []

# label 파일 하나씩 확인
for label_file in label_folder.glob("*.txt"):

    class_list = []

    # label 파일 읽기
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # label 파일 안의 class 번호 확인
    for line in lines:
        line = line.strip()

        if line == "":
            continue

        class_number = int(line.split()[0])
        class_list.append(class_number)

    # SCRATCH만 들어 있는 파일인지 확인
    if set(class_list) == {scratch_class}:
        scratch_only_files.append(label_file)

print("SCRATCH만 들어 있는 이미지 수:", len(scratch_only_files))

# 항상 같은 결과가 나오도록 random seed 설정
random.seed(42)

# SCRATCH 파일 순서를 랜덤하게 섞기
random.shuffle(scratch_only_files)

# keep_count 이후 파일들은 삭제 대상으로 설정
delete_files = scratch_only_files[keep_count:]

print("삭제할 SCRATCH 이미지 수:", len(delete_files))

# 삭제 실행
for label_file in delete_files:

    image_file = image_folder / (label_file.stem + ".jpg")

    # 이미지 파일 삭제
    if image_file.exists():
        image_file.unlink()

    # label 파일 삭제
    if label_file.exists():
        label_file.unlink()

print("SCRATCH undersampling 완료")
print("남긴 SCRATCH 이미지 수:", keep_count)