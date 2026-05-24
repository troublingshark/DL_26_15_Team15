from pathlib import Path
import shutil
# OpenCV 라이브러리를 불러오기
# -> 이미지 읽고 밝게, 어둡게, blur 처리에 사용
import cv2

# 증강 대상 이미지와 라벨 폴더
image_folder = Path("target_classes/images/train")
label_folder = Path("target_classes/labels/train")

# 증강된 이미지와 라벨을 저장할 폴더
aug_image_folder = Path("augmented_target_classes/images/train")
aug_label_folder = Path("augmented_target_classes/labels/train")

# 저장 폴더 만들기
aug_image_folder.mkdir(parents=True, exist_ok=True)
aug_label_folder.mkdir(parents=True, exist_ok=True)

# 클래스 번호
# 1: COATING BAD
# 2: PARTICLE
# 3: PIO PARTICLE
# 4: PO CONTAMINATION
# 6: SEZ BURNT

# PARTICLE과 PO CONTAMINATION은 이미 어느 정도 개수가 있으므로 일부만 증강
# PARTICLE과 PO CONTAMINATION은 너무 많이 증강하지 않으려고 개수를 따로 셈
particle_aug_count = 0
po_aug_count = 0
particle_limit = 60 # 개수제한
po_limit = 100 # 개수제한

# 전체 증강 이미지 개수
total_aug_count = 0

# 클래스별 증강 개수 확인용
coating_aug_count = 0
particle_total_count = 0
pio_aug_count = 0
po_total_count = 0
sez_aug_count = 0

# label 파일을 하나씩 확인
for label_file in label_folder.glob("*.txt"):

    # label과 같은 이름의 이미지 파일 찾기
    image_file = image_folder / (label_file.stem + ".jpg")

    if not image_file.exists():
        print("이미지를 찾지 못했습니다:", image_file.name)
        continue

    # 이미지 읽기
    image = cv2.imread(str(image_file))

    if image is None:
        print("이미지를 읽지 못했습니다:", image_file.name)
        continue

    # label 파일에서 class 번호 읽기
    class_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        class_number = int(line.split()[0])
        class_list.append(class_number)

    # 이 이미지에서 만들 증강 결과를 저장할 리스트
    augmented_images = []
    # SEZ BURNT가 포함된 이미지는 가장 많이 증강
    if 6 in class_list:
        # 이미지 밝게 만드는 코드
        img_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
        # 이미지 어둡게 만드는 코드
        img_dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
        # 이미지 대비(픽셀 값 차이 더 크게 만들어서 명암 차이 강조)높이는 코드
        img_contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        # 이미지 약간 흐리게 만드는 코드
        img_blur = cv2.GaussianBlur(image, (3, 3), 0)
        # 조금 더 밝게, 조금 더 어둡게 만드는 코드
        img_bright2 = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
        img_dark2 = cv2.convertScaleAbs(image, alpha=1.0, beta=-50)

        augmented_images.append(("sez_bright", img_bright))
        augmented_images.append(("sez_dark", img_dark))
        augmented_images.append(("sez_contrast", img_contrast))
        augmented_images.append(("sez_blur", img_blur))
        augmented_images.append(("sez_bright2", img_bright2))
        augmented_images.append(("sez_dark2", img_dark2))
        sez_aug_count += 6

    # PIO PARTICLE 이미지 밝게 만든 이미지 1장만 생성
    elif 3 in class_list:
        img_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
        augmented_images.append(("pio_bright", img_bright))
        pio_aug_count += 1

    # COATING BAD 이미지 밝게 만든 이미지 1장만 생성
    elif 1 in class_list:
        img_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
        augmented_images.append(("coating_bright", img_bright))
        coating_aug_count += 1

    # PO CONTAMINATION 증강 개수가 100장보다 적을 때만 증강
    elif 4 in class_list:
        if po_aug_count < po_limit:
            img_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
            augmented_images.append(("po_bright", img_bright))
            po_aug_count += 1
            po_total_count += 1

    # PARTICLE 최대 60장까지만 증강
    elif 2 in class_list:
        if particle_aug_count < particle_limit:
            img_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
            augmented_images.append(("particle_bright", img_bright))
            particle_aug_count += 1
            particle_total_count += 1

    # 증강 이미지와 라벨 저장
    for aug_name, aug_image in augmented_images:
        # 새 이미지,라벨 파일 이름을 만드는 코드
        new_image_name = label_file.stem + "_" + aug_name + ".jpg"
        new_label_name = label_file.stem + "_" + aug_name + ".txt"

        # 증강 이미지를 실제 jpg 파일로 저장하는 코드
        cv2.imwrite(str(aug_image_folder / new_image_name), aug_image)
        # 기존 label 파일을 새 이름으로 복사하는 코드
        shutil.copy(label_file, aug_label_folder / new_label_name)
        total_aug_count += 1

print("증강 완료")
print("전체 생성된 증강 이미지 수:", total_aug_count)
print("COATING BAD 증강 수:", coating_aug_count)
print("PARTICLE 증강 수:", particle_total_count)
print("PIO PARTICLE 증강 수:", pio_aug_count)
print("PO CONTAMINATION 증강 수:", po_total_count)
print("SEZ BURNT 증강 수:", sez_aug_count)