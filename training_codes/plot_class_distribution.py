from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# train label 폴더 경로
label_folder = Path("dataset/labels/train")

# 클래스 이름
class_names = [
    "BLOCK ETCH",
    "COATING BAD",
    "PARTICLE",
    "PIO PARTICLE",
    "PO CONTAMINATION",
    "SCRATCH",
    "SEZ BURNT"
]

# 클래스별 객체 개수를 저장할 변수
class_count = Counter()

# label 폴더 안의 txt 파일을 하나씩 확인
for label_file in label_folder.glob("*.txt"):

    # label 파일 읽기
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # label 파일 안의 각 줄 확인
    for line in lines:
        line = line.strip()

        # 빈 줄이면 넘어감
        if line == "":
            continue

        # YOLO label에서 첫 번째 값이 class 번호
        class_number = int(line.split()[0])

        # 해당 class 개수 1 증가
        class_count[class_number] += 1

# 그래프에 사용할 클래스별 개수 리스트 만들기
counts = []

for i in range(len(class_names)):
    counts.append(class_count[i])

# 클래스별 개수 출력
print("===== 클래스별 객체 개수 =====")

for i in range(len(class_names)):
    print(class_names[i], ":", counts[i])

# 막대그래프 그리기
plt.figure(figsize=(10, 5))
bars = plt.bar(class_names, counts)

# 막대 위에 개수 표시
for i in range(len(bars)):
    plt.text(
        i,
        counts[i],
        str(counts[i]),
        ha="center",
        va="bottom"
    )

# 그래프 설정
plt.ylabel("instances")
plt.xticks(rotation=90)
plt.ylim(0, max(counts) + 100)
plt.tight_layout()

# 그래프 이미지 저장
plt.savefig("class_distribution_after_aug.png", dpi=300)

# 그래프 화면에 출력
plt.show()

print("그래프 저장 완료: class_distribution_after_aug.png")