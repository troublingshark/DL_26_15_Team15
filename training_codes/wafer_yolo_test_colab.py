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

# yolov8m_e30모델에서의 best.pt파일 업로드
from google.colab import files
uploaded = files.upload()

# yolov8m_e30모델 test데이터 이용한 predict 실행
from ultralytics import YOLO
model = YOLO("/content/wafer_yolov8m_e30_best.pt")
results = model.predict(
    source="/content/dataset/images/test",
    conf=0.462, # 학습결과 mask에서 나온 최상의 confidence_threshold 반영
    imgsz=640,
    device=0,
    save=True,
    show_labels=True,
    show_conf=True,
    line_width=2,
    project="/content/predictions",
    name="yolov8m_e30_test_predictions"
)
print("test 전체 예측 완료")
print("결과 저장 위치: /content/predictions/yolov8m_e30_test_predictions")

# yolov8m_e30모델 predict 결과 압축 및 다운로드
import shutil
folder_path = "/content/predictions/yolov8m_e30_test_predictions"
zip_path = "/content/yolov8m_e30_test_predictions"
shutil.make_archive(zip_path, "zip", folder_path)
print("압축 완료:", zip_path + ".zip")
from google.colab import files
files.download("/content/yolov8m_e30_test_predictions.zip")

# yolov8m_e30모델 test데이터 이용한 모델 최종성능 평가
from ultralytics import YOLO
model = YOLO("/content/wafer_yolov8m_e30_best.pt")
metrics = model.val(
    data="/content/dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    name="yolov8m_e30_test_eval"
)
print("test 데이터 최종 평가 완료")

# yolov8m_e30모델 최종성능 평가결과 압축 및 다운로드
import shutil
folder_path = "/content/runs/segment/yolov8m_e30_test_eval"
zip_path = "/content/yolov8m_e30_test_eval"
shutil.make_archive(zip_path, "zip", folder_path)
print("압축 완료:", zip_path + ".zip")
from google.colab import files
files.download("/content/yolov8m_e30_test_eval.zip")

#-------------------------------------------------------------------

# yolov8m_e50모델에서의 best.pt파일 업로드
from google.colab import files
uploaded = files.upload()

# yolov8m_e50모델 test데이터 이용한 predict 실행
from ultralytics import YOLO
model = YOLO("/content/wafer_yolov8m_e50_best.pt")
results = model.predict(
    source="/content/dataset/images/test",
    conf=0.432, # 학습결과 mask에서 나온 최상의 confidence_threshold 반영
    imgsz=640,
    device=0,
    save=True,
    show_labels=True,
    show_conf=True,
    line_width=2,
    project="/content/predictions",
    name="yolov8m_e50_test_predictions"
)
print("test 전체 예측 완료")
print("결과 저장 위치: /content/predictions/yolov8m_e50_test_predictions")


# yolov8m_e50모델 test데이터 이용한 모델 최종성능 평가
from ultralytics import YOLO
model = YOLO("/content/wafer_yolov8m_e50_best.pt")
metrics = model.val(
    data="/content/dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    name="yolov8m_e50_test_eval"
)
print("test 데이터 최종 평가 완료")

#-------------------------------------------------------------------

# yolov8m_e100모델에서의 best.pt파일 업로드
from google.colab import files
uploaded = files.upload()

# yolov8m_e100모델 test데이터 이용한 predict 실행
from ultralytics import YOLO
model = YOLO("/content/wafer_yolov8m_e100_best.pt")
results = model.predict(
    source="/content/dataset/images/test",
    conf=0.434, # 학습결과 mask에서 나온 최상의 confidence_threshold 반영
    imgsz=640,
    device=0,
    save=True,
    show_labels=True,
    show_conf=True,
    line_width=2,
    project="/content/predictions",
    name="yolov8m_e100_test_predictions"
)
print("test 전체 예측 완료")
print("결과 저장 위치: /content/predictions/yolov8m_e100_test_predictions")


# yolov8m_e100모델 test데이터 이용한 모델 최종성능 평가
from ultralytics import YOLO
model = YOLO("/content/wafer_yolov8m_e100_best.pt")
metrics = model.val(
    data="/content/dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    name="yolov8m_e100_test_eval"
)
print("test 데이터 최종 평가 완료")

#-------------------------------------------------------------------

# yolov8m_aug_img640_e80모델의 test데이터 성능평가
from ultralytics import YOLO
model_path = "/content/runs/segment/yolov8m_aug_img640_e80/weights/best.pt"
data_path = "/content/dataset/data.yaml"

model = YOLO(model_path)
results = model.val(
    data=data_path,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    name="yolov8m_aug_img640_e80_test_eval"
)
print("test 데이터 최종 평가 완료")

#-------------------------------------------------------------------

# yolov8m_aug_img1024_e80모델의 test데이터 성능평가
from ultralytics import YOLO
model_path = "/content/runs/segment/yolov8m_aug_img1024_e80/weights/best.pt"
data_path = "/content/dataset/data.yaml"

model = YOLO(model_path)
results = model.val(
    data=data_path,
    split="test",
    imgsz=1024,
    batch=8,
    device=0,
    name="yolov8m_aug_img1024_e80_test_eval"
)
print("test 데이터 최종 평가 완료")

#-------------------------------------------------------------------
