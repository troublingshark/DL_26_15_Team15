from pathlib import Path

# 데이터셋 경로 설정
image_folder = Path("dataset_balanced/images/train")
label_folder = Path("dataset_balanced/labels/train")

# 이미지 파일과 라벨 파일 불러오기
image_files = list(image_folder.glob("*.jpg"))
label_files = list(label_folder.glob("*.txt"))

# 파일 개수 확인
print("===== 데이터셋 파일 개수 확인 =====")
print("train 이미지 수:", len(image_files))
print("train 라벨 수:", len(label_files))


# 확장자를 제외한 파일 이름 추출
image_names = set()

for image_file in image_files:
    image_names.add(image_file.stem)

label_names = set()

for label_file in label_files:
    label_names.add(label_file.stem)

# 이미지와 라벨 매칭 확인
only_images = image_names - label_names
only_labels = label_names - image_names

print("\n===== 이미지-라벨 매칭 확인 =====")
print("라벨이 없는 이미지 수:", len(only_images))
print("이미지가 없는 라벨 수:", len(only_labels))

# 최종 결과 출력
if len(only_images) == 0 and len(only_labels) == 0:
    print("\n결과: 모든 이미지와 라벨 파일이 정상적으로 매칭됩니다.")
else:
    print("\n결과: 매칭되지 않는 파일이 있습니다.")