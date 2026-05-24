from pathlib import Path
import shutil

# 원본 이미지와 라벨 폴더
image_folder = Path("dataset/images/train")
label_folder = Path("dataset/labels/train")

# 증강할 대상 이미지를 따로 모아둘 폴더
target_image_folder = Path("target_classes/images/train")
target_label_folder = Path("target_classes/labels/train")

# 이미지,라벨을 저장할 저장폴더 없으면 새로 만들기
target_image_folder.mkdir(parents=True, exist_ok=True)
target_label_folder.mkdir(parents=True, exist_ok=True)

# 증강할 클래스 번호
# 1: COATING BAD
# 2: PARTICLE
# 3: PIO PARTICLE
# 4: PO CONTAMINATION
# 6: SEZ BURNT
target_classes = [1, 2, 3, 4, 6]

# 복사한 이미지 개수
copy_count = 0

# train label 폴더 안의 txt 파일을 하나씩 확인
for label_file in label_folder.glob("*.txt"):

    # 이 label 파일에 증강 대상 클래스가 있는지 확인하는 변수
    has_target_class = False

    # label 파일 열기
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # label 파일 안의 각 줄을 하나씩 확인
    for line in lines:
        line = line.strip()

        # 빈 줄이면 넘어감
        if line == "":
            continue

        # YOLO label에서 첫 번째 값이 class 번호
        class_number = int(line.split()[0])

        # 현재 class가 증강 대상 클래스이면 표시
        if class_number in target_classes:
            has_target_class = True
            break

    # 증강 대상 클래스가 들어 있는 경우 이미지와 라벨 복사
    if has_target_class: # 현재 label 파일 안에 증강 대상 클래스가 들어 있다면
        # 라벨 파일과 같은 이름의 이미지 파일을 찾는 코드
        image_file = image_folder / (label_file.stem + ".jpg")

        if image_file.exists():
            # 이미지,라벨 파일을 폴더로 복사하는 코드
            shutil.copy(image_file, target_image_folder / image_file.name)
            shutil.copy(label_file, target_label_folder / label_file.name)
            copy_count += 1
        else:
            print("이미지 파일을 찾지 못했습니다:", image_file.name)

print("증강 대상 클래스 이미지 추출 완료")
print("복사된 이미지 수:", copy_count)