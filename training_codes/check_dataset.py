from pathlib import Path

# 데이터셋 폴더 경로
dataset_path = Path("dataset")

# 검사할 폴더
splits = ["train", "valid", "test"]

print("=" * 60)
print("YOLO 데이터셋 구조 검사 시작")
print("=" * 60)

for split in splits:
    image_path = dataset_path / "images" / split
    label_path = dataset_path / "labels" / split

    print(f"\n[{split}] 검사")

    # jpg 이미지 파일과 txt 라벨 파일 가져오기
    image_files = list(image_path.glob("*.jpg"))
    label_files = list(label_path.glob("*.txt"))
    
    image_names = set([file.stem for file in image_files])
    label_names = set([file.stem for file in label_files])

    # 서로 매칭되지 않는 파일 찾기
    no_label_images = image_names - label_names
    no_image_labels = label_names - image_names

    print(f"이미지 개수: {len(image_files)}")
    print(f"라벨 개수: {len(label_files)}")
    print("라벨 없는 이미지:", len(no_label_images))
    print("이미지 없는 라벨:", len(no_image_labels))

print("\n" + "=" * 60)
print("검사 완료")
print("=" * 60)