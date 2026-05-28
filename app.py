"""
Wafer Defect Analysis Service
YOLOv8 Segmentation 모델로 여러 웨이퍼 이미지의 결함을 자동 검출하고,
표·그래프·CSV·ZIP·TXT 보고서로 결과를 제공합니다.
"""

from pathlib import Path
from io import BytesIO # 결과 이미지들 ZIP으로 다운로드할때 사용
import datetime
import zipfile # 여러 결과 이미지 ZIP 파일로 묶을 때 사용

import cv2 # 이미지 색상 순서 바꾸는데 사용(BGR -> RGB)
import numpy as np
import pandas as pd
import streamlit as st # 웹 서비스 화면을 만드는 핵심 라이브러리
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 0. 기본 설정값
# ============================================================
MODEL_PATH = Path("models/best.pt")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIDENCE = 0.25 # confidence_threshold: 기본 0.25로 설정
UPLOAD_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"] # 업로드
DELETE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"} # 초기화


# ============================================================
# 1. 모델 로드
# ============================================================
@st.cache_resource # 처음 실행할때 best.pt 한번만 로드 -> 그다음부터는 캐시에 저장된 모델 재사용
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    return YOLO(str(MODEL_PATH)) # Path 객체를 문자열로 변환


# ============================================================
# 2. YOLO 추론 및 결과 처리 함수
# ============================================================
def run_inference(model: YOLO, image_np: np.ndarray, confidence: float):
    # 이미지 1장을 YOLO 모델에 넣어서 결함 검출 함수 -> confidence 값보다 낮은 검출 결과는 YOLO 내부에서 자동으로 제외

    results = model.predict(
        source=image_np,
        task="segment",  # detection이 아니라 segmentation 수행
        conf=confidence, # confidence: 사용자가 설정한 confidence_threshold
        verbose=False,
    )
    return results[0] # 이미지 1장에 대한 결과 반환


def draw_result_image(result) -> Image.Image:
    # YOLO 결과에 mask, box, label을 그린 이미지를 PIL Image로 반환

    annotated_bgr = result.plot() # .plot() 기능 -> 자동으로 mask, box, label, con이 그려진 이미지 만들어줌
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB) # 색상순서: bgr -> rgb로 변환(PIL/Streamlit RGB 사용)
    return Image.fromarray(annotated_rgb) # numpy 배열 이미지 -> PIL Image로 바꿔서 반환


def save_result_image(result_image: Image.Image, original_name: str) -> Path:
    # YOLO 결과 이미지(mask 그려진 이미지) outputs 폴더에 저장

    stem = Path(original_name).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = OUTPUT_DIR / f"{stem}_{timestamp}_result.png"
    result_image.save(save_path)
    return save_path


def extract_detection_rows(result, image_name: str, result_image_path: str) -> list[dict]:
    # YOLO 결과에서 결함 정보를 꺼내 DataFrame 행 형태의 딕셔너리 목록으로 반환

    boxes = result.boxes
    masks = result.masks
    class_names = result.names

    if boxes is None or len(boxes) == 0: # 결함이 없으면 빈 리스트 반환
        return []

    rows = []

    for idx, box in enumerate(boxes): # 검출결함:3개 -> 반복문:3번 반복
        class_id = int(box.cls.item()) # class 번호
        confidence = float(box.conf.item()) # 해당 결함 confidence 값

        mask_area = 0
        if masks is not None and idx < len(masks.data):
            mask_area = int(masks.data[idx].sum().item())

        rows.append(
            {
                "image_name": image_name,
                "defect_class": class_names[class_id],
                "confidence": round(confidence, 4),
                "mask_area": mask_area,
                "result_image_path": result_image_path,
            }
        )
    return rows


# ============================================================
# 3. 표, 그래프, 보고서에 사용할 분석 함수
# ============================================================
def build_class_summary(full_df: pd.DataFrame) -> pd.DataFrame:
    # 전체 검출 결과 -> class별로 묶어 요약표 만듬

    total_count = len(full_df) # 전체 결함 개수
    summary = (
        full_df.groupby("defect_class") # defect_class 기준으로 묶기
        .agg( # 각 class마다 통계 계산
            count=("confidence", "count"),
            avg_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
            total_mask_area=("mask_area", "sum"),
        )
        .reset_index()
    )

    # 각 class 전체 결함 중 몇 %(비율) 계산
    summary["ratio_percent"] = (summary["count"] / total_count * 100).round(2)
    summary["avg_confidence"] = summary["avg_confidence"].round(4)
    summary["min_confidence"] = summary["min_confidence"].round(4)
    summary["max_confidence"] = summary["max_confidence"].round(4)

    summary = summary[
        [
            "defect_class",
            "count",
            "ratio_percent",
            "avg_confidence",
            "min_confidence",
            "max_confidence",
            "total_mask_area",
        ]
    ]
    # 많이 나온 결함 순으로 정렬
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


def build_auto_summary(num_images: int, full_df: pd.DataFrame, class_summary: pd.DataFrame) -> str:
    # 화면에 보여줄 간단한 분석 요약 문장 생성

    total_defects = len(full_df)
    most_row = class_summary.iloc[0] # 이미 내림차순 정렬 되어있음 -> 가장 많이 나온 클래스
    least_row = class_summary.iloc[-1] # 가장 적게 나온 클래스
    low_conf_row = class_summary.loc[class_summary["avg_confidence"].idxmin()] # 평균 confidence가 가장 낮은 class 행 찾기
    return "\n".join(
        [
            f"- 총 업로드 이미지 수: **{num_images}장**",
            f"- 총 검출 결함 수: **{total_defects}개**",
            f"- 가장 많이 검출된 결함: **{most_row['defect_class']}** ({int(most_row['count'])}개)",
            f"- 가장 적게 검출된 결함: **{least_row['defect_class']}** ({int(least_row['count'])}개)",
            f"- 평균 confidence가 가장 낮은 결함: **{low_conf_row['defect_class']}** "
            f"(avg = {low_conf_row['avg_confidence']:.4f})",
            f"- ⚠️ **{low_conf_row['defect_class']}** 결함은 confidence가 낮으므로 "
            "데이터 보강 또는 추가 학습이 필요할 수 있습니다.",
        ]
    )


def build_report_text(
    num_images: int,
    full_df: pd.DataFrame,
    class_summary: pd.DataFrame,
    confidence_threshold: float,
) -> str:
    # 분석 결과를 바탕으로 TXT 보고서 내용 생성
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_defects = len(full_df)

    most_row = class_summary.iloc[0]
    least_row = class_summary.iloc[-1]
    low_conf_row = class_summary.loc[class_summary["avg_confidence"].idxmin()]

    ratio_lines = []
    for _, row in class_summary.iterrows():
        ratio_lines.append(
            f"  - {row['defect_class']}: {int(row['count'])}개 ({row['ratio_percent']:.2f}%)"
        )
    ratio_summary = "\n".join(ratio_lines)

    return f"""\
========================================
  Wafer Defect Analysis Report
  생성 일시: {now}
========================================

[1. 분석 개요]
  - 총 업로드 이미지 수        : {num_images}장
  - 적용 Confidence Threshold : {confidence_threshold:.2f}
  - 총 검출 결함 수            : {total_defects}개

[2. 결함 class별 비율 요약]
{ratio_summary}

[3. 결함 검출 현황]
  - 가장 많이 검출된 결함 : {most_row['defect_class']} ({int(most_row['count'])}개)
  - 가장 적게 검출된 결함 : {least_row['defect_class']} ({int(least_row['count'])}개)

[4. Confidence 분석]
  - 평균 confidence가 가장 낮은 결함 : {low_conf_row['defect_class']} (avg = {low_conf_row['avg_confidence']:.4f})
  - 해석 : '{low_conf_row['defect_class']}' 결함은 모델의 예측 신뢰도가 상대적으로 낮습니다.
           추가 데이터 확보 또는 해당 class에 대한 재학습을 통해
           검출 안정성을 개선할 필요가 있을 수 있습니다.

[5. 서비스 안내]
  이 서비스는 업로드된 웨이퍼 이미지에 대한 결함 분석 서비스입니다.
  mAP50, mAP50-95와 같은 모델 성능 평가는 정답 label과 예측 결과를 비교해야 하므로
  본 서비스 범위에는 포함하지 않았습니다.

========================================
  End of Report
========================================
"""


# ============================================================
# 4. 다운로드 및 초기화 보조 함수
# ============================================================
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    # DataFrame을 한글이 깨지지 않는 CSV bytes로 변환
    return df.to_csv(index=False).encode("utf-8-sig")


def make_zip_bytes(image_paths: list[str]) -> bytes:
    # 결과 이미지 여러 장 -> ZIP 파일로 묶기

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for image_path in image_paths:
            image_path = Path(image_path)
            if image_path.exists():
                zip_file.write(image_path, arcname=image_path.name)

    zip_buffer.seek(0)
    return zip_buffer.read()


def clear_output_images() -> int:
    # outputs 폴더 안의 결과 이미지들을 삭제하는 함수

    deleted_count = 0
    for file_path in OUTPUT_DIR.iterdir():
        # outputs 폴더 안의 파일들 하나씩 확인
        if file_path.is_file() and file_path.suffix.lower() in DELETE_IMAGE_EXTENSIONS:
            file_path.unlink()
            deleted_count += 1
    return deleted_count


# ============================================================
# 5. Streamlit 화면 구성 함수
# ============================================================
def initialize_session_state() -> None:
    # 파일 업로더 초기화에 필요한 session_state 값 준비
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "reset_done" not in st.session_state:
        st.session_state.reset_done = False


def render_sidebar() -> tuple[float, bool]:
    # 사이드바를 그리고 confidence threshold와 이미지 표시 옵션 반환
    with st.sidebar:
        st.header("사용 방법")
        st.markdown(
            "1. Confidence Threshold를 설정합니다.\n"
            "2. 웨이퍼 이미지를 여러 장 업로드합니다.\n"
            "3. 결함 검출 결과와 통계 그래프를 확인합니다.\n"
            "4. CSV, ZIP, TXT 보고서를 다운로드합니다."
        )

        st.divider()

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
        )

        st.markdown(f"**현재 적용 값: `{confidence_threshold:.2f}`**")
        st.info(
            "Threshold가 낮을수록 더 많은 결함 후보가 검출되고, "
            "높을수록 신뢰도가 높은 결함만 표시됩니다."
        )

        show_image_detail = st.checkbox(
            "이미지별 원본/결과 이미지 표시",
            value=False,
            help="이미지가 많을 때는 끄는 것을 권장합니다.",
        )

        st.divider()
        st.caption("Model: YOLOv8-seg")
        st.caption(f"Path: {MODEL_PATH}")

        st.divider()

        if st.button("🗑️ 결과 초기화", use_container_width=True, type="primary"):
            clear_output_images()
            st.session_state.uploader_key += 1
            st.session_state.reset_done = True
            st.rerun()
    return confidence_threshold, show_image_detail


def analyze_uploaded_images(uploaded_files: list, model: YOLO, confidence_threshold: float, show_image_detail: bool) -> pd.DataFrame:
    # 업로드된 모든 이미지를 분석 -> 전체 검출 결과 DataFrame 반환

    all_rows = [] # 모든 이미지에서 나온 결함 정보 저장 리스트
    progress_bar = st.progress(0) # 이미지가 여러 장일 때 분석 진행률 표시

    for idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((idx + 1) / len(uploaded_files))

        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image) # 업로드된 이미지를 PIL로 열고, RGB로 통일 -> numpy 배열로 변환

        with st.spinner(f"{uploaded_file.name} 분석 중..."):
            result = run_inference(model, image_np, confidence_threshold) # 이미지 1장 YOLO 모델에 넣어서 결과 추출

        result_image = draw_result_image(result)
        saved_path = save_result_image(result_image, uploaded_file.name) # mask가 그려진 결과 이미지 만듬 -> outputs 폴더에 저장해.

        if show_image_detail:
            # 체크박스가 켜져 있을 때만 원본/결과 이미지  화면에 표시
            with st.expander(f"[{idx + 1}/{len(uploaded_files)}] {uploaded_file.name}", expanded=False):
                col_original, col_result = st.columns(2)

                with col_original:
                    st.caption("원본 이미지")
                    st.image(pil_image, use_container_width=True)

                with col_result:
                    st.caption("검출 결과 이미지")
                    st.image(result_image, use_container_width=True)

                st.caption(f"결과 저장 위치: `{saved_path}`")

        rows = extract_detection_rows(result, uploaded_file.name, str(saved_path))
        all_rows.extend(rows)

    progress_bar.empty()
    return pd.DataFrame(all_rows)


def render_download_buttons(full_df: pd.DataFrame, class_summary: pd.DataFrame, report_text: str) -> None:
    # CSV, ZIP, TXT 다운로드 버튼 한 곳에 모아 표시
    st.markdown("---")
    st.subheader("결과 다운로드")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="전체 검출 결과 CSV",
            data=dataframe_to_csv_bytes(full_df),
            file_name="detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="class별 요약 CSV",
            data=dataframe_to_csv_bytes(class_summary),
            file_name="class_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col3:
        result_image_paths = full_df["result_image_path"].unique().tolist()
        st.download_button(
            label="결과 이미지 ZIP",
            data=make_zip_bytes(result_image_paths),
            file_name="wafer_result_images.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with col4:
        st.download_button(
            label="분석 보고서 TXT",
            data=report_text.encode("utf-8-sig"),
            file_name="wafer_defect_report.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
        )


# ============================================================
# 6. 메인 실행 함수
# ============================================================
def main() -> None:
    st.set_page_config(
        page_title="Wafer Defect Analysis",
        page_icon="🔬",
        layout="wide",
    )

    initialize_session_state()

    st.title("🔬 Wafer Defect Analysis Service")
    st.markdown("YOLOv8 Segmentation 모델로 웨이퍼 이미지의 결함을 자동으로 검출합니다.")

    if st.session_state.reset_done:
        st.success("✅ 분석 결과와 업로드 파일 목록이 초기화되었습니다.")
        st.session_state.reset_done = False

    confidence_threshold, show_image_detail = render_sidebar()

    try:
        with st.spinner("모델을 불러오는 중..."):
            model = load_model()
    except Exception as error:
        st.error(f"모델 로드 중 오류가 발생했습니다: {error}")
        st.stop()

    st.markdown(
        f"> 현재 Confidence Threshold: **{confidence_threshold:.2f}**  \n"
        "> 이 값 미만의 검출 결과는 분석에서 제외됩니다."
    )

    uploaded_files = st.file_uploader(
        "웨이퍼 이미지 업로드 (여러 장 선택 가능)",
        type=UPLOAD_IMAGE_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if not uploaded_files:
        st.info("이미지를 업로드하면 분석이 시작됩니다.")
        return

    st.markdown(f"### 총 {len(uploaded_files)}장 이미지 분석 결과")

    full_df = analyze_uploaded_images(
        uploaded_files=uploaded_files,
        model=model,
        confidence_threshold=confidence_threshold,
        show_image_detail=show_image_detail,
    )

    if full_df.empty:
        st.warning("현재 threshold 기준으로 검출된 결함이 없습니다. threshold를 낮춰 다시 시도해 보세요.")
        return

    full_df = full_df[
        ["image_name", "defect_class", "confidence", "mask_area", "result_image_path"]
    ]

    class_summary = build_class_summary(full_df)
    report_text = build_report_text(
        num_images=len(uploaded_files),
        full_df=full_df,
        class_summary=class_summary,
        confidence_threshold=confidence_threshold,
    )

    st.markdown("---")
    st.subheader("전체 검출 결과")
    st.dataframe(
        full_df,
        use_container_width=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                label="confidence",
                format="%.4f",
                min_value=0.0,
                max_value=1.0,
            )
        },
    )

    st.markdown("---")
    st.subheader("class별 요약")
    st.dataframe(class_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("결함 통계 그래프")

    chart_df = class_summary.set_index("defect_class")
    col_count, col_ratio, col_confidence = st.columns(3)

    with col_count:
        st.markdown("**class별 결함 개수**")
        st.bar_chart(chart_df[["count"]])

    with col_ratio:
        st.markdown("**class별 비율 (%)**")
        st.bar_chart(chart_df[["ratio_percent"]])

    with col_confidence:
        st.markdown("**class별 평균 confidence**")
        st.bar_chart(chart_df[["avg_confidence"]])

    st.markdown("---")
    st.subheader("분석 요약")
    st.markdown(build_auto_summary(len(uploaded_files), full_df, class_summary))

    st.markdown("---")
    st.subheader("자동 분석 보고서")
    st.text_area("보고서 내용", value=report_text, height=400)

    render_download_buttons(full_df, class_summary, report_text)


if __name__ == "__main__":
    main()
