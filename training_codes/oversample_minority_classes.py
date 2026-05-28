from pathlib import Path
import shutil
import random

# 폴더 경로 설정
# 소수 클래스 이미지와 라벨이 모여 있는 폴더
minority_image_folder = Path("minority_from_balanced/images/train")
minority_label_folder = Path("minority_from_balanced/labels/train")

# oversampling 결과를 추가할 dataset_balanced train 폴더
train_image_folder = Path("dataset_balanced/images/train")
train_label_folder = Path("dataset_balanced/labels/train")


# 클래스 정보 설정
# oversampling할 클래스 이름
class_names = {
    1: "COATING BAD",
    2: "PARTICLE",
    3: "PIO PARTICLE",
    4: "PO CONTAMINATION",
    6: "SEZ BURNT"
}

# 클래스별로 추가할 이미지 수
add_counts = {
    1: 288,
    2: 139,
    3: 299,
    4: 183,
    6: 506
}

# 항상 같은 결과가 나오도록 random seed 설정
random.seed(42)


# 클래스별 label 파일 저장 공간 만들기
class_files = {
    1: [],
    2: [],
    3: [],
    4: [],
    6: []
}


# label 파일을 읽어서 클래스별로 분류
for label_file in minority_label_folder.glob("*.txt"):

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 첫 번째 줄에서 class 번호 가져오기
    # 첫 번째 class 번호를 대표 class로 사용
    first_line = lines[0].strip()

    if first_line == "":
        continue

    class_number = int(first_line.split()[0])

    # oversampling 대상 클래스이면 해당 클래스 목록에 저장
    if class_number in class_files:
        class_files[class_number].append(label_file)



# 5. 클래스별 oversampling 수행
total_copy_count = 0

for class_number in add_counts:

    files = class_files[class_number]
    add_count = add_counts[class_number]

    print("\n클래스:", class_names[class_number])
    print("원본 후보 이미지 수:", len(files))
    print("추가할 이미지 수:", add_count)

    if len(files) == 0:
        print("해당 클래스 파일이 없어 건너뜁니다.")
        continue

    # 파일 순서를 섞어서 특정 파일만 앞에서 반복되지 않도록 함
    random.shuffle(files)

    # 필요한 개수만큼 복사본 만들기
    for i in range(add_count):

        # 후보 파일들을 반복해서 사용
        label_file = files[i % len(files)]
        image_file = minority_image_folder / (label_file.stem + ".jpg")

        if not image_file.exists():
            print("이미지를 찾지 못했습니다:", image_file.name)
            continue

        # 새 파일 이름 만들기
        new_stem = label_file.stem + "_over_" + str(class_number) + "_" + str(i)

        new_image_file = train_image_folder / (new_stem + ".jpg")
        new_label_file = train_label_folder / (new_stem + ".txt")

        # 이미지와 라벨 파일 복사
        shutil.copy(image_file, new_image_file)
        shutil.copy(label_file, new_label_file)

        total_copy_count += 1


# 최종 결과 출력
print("\n소수 클래스 oversampling 완료")
print("추가된 이미지 수:", total_copy_count)