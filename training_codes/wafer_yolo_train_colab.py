# YOLOv8을 사용할 때 필요한 라이브러리 설치
!pip install ultralytics

# 현재 환경에서 CUDA GPU를 사용할 수 있는지 확인
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 사용 불가")

# dataset 압축 파일 업로드
from google.colab import files
uploaded = files.upload()

#dataset 압축 풀기
import zipfile
import os
zip_path = "/content/dataset.zip.zip"
extract_path = "/content"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("압축 해제 완료")

# dataset의 train/val/test에서 이미지,라벨 매칭 여부 검사
from pathlib import Path
dataset_dir = Path("/content/dataset")
for split in ["train", "valid", "test"]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    image_files = list(image_dir.glob("*.jpg"))
    label_files = list(label_dir.glob("*.txt"))
    print(f"[{split}]")
    print("이미지 개수:", len(image_files))
    print("라벨 개수:", len(label_files))
    print()

#-------------------------------------------------------------------

# yolov8n-seg모델 30 epoch 학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "/content/dataset/data.yaml"
model = YOLO("yolov8n-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=8,
    device=0,
    name="wafer_yolov8n_e30",
    patience=10
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

# yolov8n-seg모델 30 epoch 학습결과 압축 및 다운로드
import shutil
folder_path = "/content/runs/segment/wafer_yolov8n_e30"
zip_path = "/content/wafer_yolov8n_e30_result"
shutil.make_archive(zip_path, "zip", folder_path)
print("압축 완료:", zip_path + ".zip")
from google.colab import files
files.download("/content/wafer_yolov8n_e30_result.zip")

#-------------------------------------------------------------------

# yolov8s-seg모델 30 epoch 학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "/content/dataset/data.yaml"
model = YOLO("yolov8s-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=8,
    device=0,
    name="wafer_yolov8s_e30",
    patience=10
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

#-------------------------------------------------------------------

# yolov8m-seg모델 30 epoch 학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "/content/dataset/data.yaml"
model = YOLO("yolov8m-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=8,
    device=0,
    name="wafer_yolov8m_e30",
    patience=10
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

#-------------------------------------------------------------------

# yolov8m-seg모델 50 epoch 학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "/content/dataset/data.yaml"
model = YOLO("yolov8m-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    name="wafer_yolov8m_e50",
    patience=15
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

#-------------------------------------------------------------------

# yolov8m-seg모델 100 epoch 학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "/content/dataset/data.yaml"
model = YOLO("yolov8m-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    name="wafer_yolov8m_e100",
    patience=20
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

#-------------------------------------------------------------------

# 데이터셋 분포 불균형 해결후 yolov8m-seg모델 img=640로 80 epoch 재학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "dataset/data.yaml"
model = YOLO("yolov8m-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=80,
    imgsz=640,
    batch=8,
    device=0,
    name="yolov8m_aug_img640_e80",
    patience=15
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

#-------------------------------------------------------------------

# 데이터셋 분포 불균형 해결후 yolov8m-seg모델 img=1024로 80 epoch 재학습시키기
from ultralytics import YOLO
import time
DATA_YAML = "dataset/data.yaml"
model = YOLO("yolov8m-seg.pt")
start_time = time.time()
model.train(
    data=DATA_YAML,
    epochs=80,
    imgsz=1024,
    batch=8,
    device=0,
    name="yolov8m_aug_img1024_e80",
    patience=15
)
end_time = time.time()
print("=" * 50)
print("학습 완료")
print(f"총 학습 시간: {(end_time - start_time) / 60:.2f}분")
print("=" * 50)

#-------------------------------------------------------------------