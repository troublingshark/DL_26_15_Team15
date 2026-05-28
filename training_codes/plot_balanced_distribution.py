from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# label 폴더 경로 설정
label_folder = Path("dataset_balanced/labels/train")

# 클래스 이름 설정
class_names = [
    "BLOCK ETCH",
    "COATING BAD",
    "PARTICLE",
    "PIO PARTICLE",
    "PO CONTAMINATION",
    "SCRATCH",
    "SEZ BURNT"
]

# 클래스별 객체 개수 세기
class_count = Counter()

for label_file in label_folder.glob("*.txt"):

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 빈 줄은 건너뛰기
        if line == "":
            continue

        # YOLO 라벨에서 첫 번째 값이 클래스 번호
        class_number = int(line.split()[0])

        # 해당 클래스 개수 증가
        class_count[class_number] += 1


# 그래프에 사용할 데이터 만들기
counts = []

for i in range(len(class_names)):
    counts.append(class_count[i])


# 5. 클래스별 개수 출력
print("===== 클래스별 객체 개수 =====")

for i in range(len(class_names)):
    print(class_names[i], ":", counts[i])

print("전체 객체 수:", sum(counts))


# 막대그래프 그리기
plt.figure(figsize=(10, 5))
bars = plt.bar(class_names, counts)

# 막대 위에 숫자 표시
for i in range(len(bars)):
    plt.text(
        i,
        counts[i],
        str(counts[i]),
        ha="center",
        va="bottom"
    )

plt.title("Class Distribution after SCRATCH Undersampling")
plt.xlabel("Class")
plt.ylabel("Instances")
plt.xticks(rotation=90)
plt.ylim(0, max(counts) + 100)
plt.tight_layout()


# 그래프 저장 및 출력
save_path = "balanced_distribution_after_scratch_under.png"
plt.savefig(save_path, dpi=300)
plt.show()
print("그래프 저장 완료:", save_path)