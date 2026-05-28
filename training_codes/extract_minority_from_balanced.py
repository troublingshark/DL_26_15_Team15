from pathlib import Path
import shutil

image_folder = Path("dataset_balanced/images/train")
label_folder = Path("dataset_balanced/labels/train")

# 소수 클래스 이미지를 따로 저장할 폴더
save_image_folder = Path("minority_from_balanced/images/train")
save_label_folder = Path("minority_from_balanced/labels/train")

# 저장 폴더 만들기
save_image_folder.mkdir(parents=True, exist_ok=True)
save_label_folder.mkdir(parents=True, exist_ok=True)

# oversampling 대상 클래스 번호
# 1: COATING BAD
# 2: PARTICLE
# 3: PIO PARTICLE
# 4: PO CONTAMINATION
# 6: SEZ BURNT
target_classes = [1, 2, 3, 4, 6]

copy_count = 0

# label 파일 하나씩 확인
for label_file in label_folder.glob("*.txt"):

    has_target_class = False

    # label 파일 읽기
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # label 안에 target class가 있는지 확인
    for line in lines:
        line = line.strip()

        if line == "":
            continue

        class_number = int(line.split()[0])

        if class_number in target_classes:
            has_target_class = True
            break

    # target class가 있으면 이미지와 label 복사
    if has_target_class:
        image_file = image_folder / (label_file.stem + ".jpg")

        if image_file.exists():
            shutil.copy(image_file, save_image_folder / image_file.name)
            shutil.copy(label_file, save_label_folder / label_file.name)
            copy_count += 1
        else:
            print("이미지를 찾지 못했습니다:", image_file.name)

print("소수 클래스 이미지 추출 완료")
print("복사된 이미지 수:", copy_count)