# Meaning-Based Image Classification (Hybrid Model)

이 프로젝트는 **MobileNetV3** (시각적 특징)와 **CLIP** (의미적 특징)을 결합한 하이브리드 이미지 분류기입니다. 이미지가 어떻게 생겼는지뿐만 아니라, 이미지가 무엇을 의미하는지까지 고려하여 분류 성능을 향상시킵니다.

## 📌 주요 기능

- **하이브리드 아키텍처**: CNN 기반의 MobileNetV3와 Transformer 기반의 CLIP 모델을 앙상블하여 사용합니다.
- **유연한 모드 선택**:
  - `hybrid`: MobileNetV3 + CLIP (권장)
  - `mobilenet`: MobileNetV3만 사용 (기존 CNN 방식)
  - `clip`: CLIP만 사용 (텍스트-이미지 의미적 연결 활용)
- **자동화된 파이프라인**: 데이터 로드, 전처리, 학습, 검증, 평가까지의 전체 과정을 지원합니다.

## 🛠️ 설치 방법 (Installation)

필요한 Python 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

## 데이터셋 준비 (Data Preparation)

데이터셋은 다음과 같은 폴더 구조로 준비해야 합니다. 각 폴더의 이름이 클래스(레이블) 이름이 됩니다.

```
dataset/
├── dog/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── cat/
│   ├── image1.jpg
│   └── ...
└── car/
    └── ...
```

## 🚀 사용 방법 (Usage)

### 1. 모델 학습 (Training)

`src/train.py` 스크립트를 사용하여 모델을 학습시킵니다.

**기본 사용법:**
```bash
python src/train.py --data_dir ./path/to/dataset
```

**주요 옵션:**
- `--data_dir`: (필수) 데이터셋이 위치한 루트 디렉토리 경로.
- `--epochs`: 학습 에폭 수 (기본값: 10).
- `--batch_size`: 배치 크기 (기본값: 32).
- `--lr`: 학습률 (기본값: 0.001).
- `--mode`: 모델 모드 선택 (`hybrid`, `mobilenet`, `clip` 중 택1, 기본값: `hybrid`).
- `--output_dir`: 체크포인트 및 로그 저장 경로 (기본값: `checkpoints`).

**예시:**
```bash
# 하이브리드 모드로 20 에폭 학습
python src/train.py --data_dir ./data/my_dataset --epochs 20 --mode hybrid --output_dir ./results
```

### 2. 모델 평가 (Evaluation)

학습된 체크포인트를 사용하여 모델의 성능을 평가합니다. `src/evaluate.py`를 실행하면 분류 보고서(Classification Report)와 혼동 행렬(Confusion Matrix) 이미지가 생성됩니다.

**사용법:**
```bash
python src/evaluate.py --data_dir ./path/to/dataset --checkpoint ./checkpoints/checkpoint_hybrid_epoch_10.pth.tar
```

**주요 옵션:**
- `--data_dir`: (필수) 평가할 데이터셋 경로 (학습 데이터와 동일한 구조).
- `--checkpoint`: (필수) 학습된 모델의 체크포인트 파일 경로 (.pth.tar).
- `--batch_size`: 배치 크기 (기본값: 32).

## 📂 파일 구조

```
.
├── requirements.txt    # 의존성 패키지 목록
├── README.md           # 프로젝트 설명서
└── src/
    ├── dataset.py      # 데이터셋 로드 및 전처리 클래스
    ├── model.py        # HybridClassifier 모델 정의
    ├── train.py        # 학습 스크립트
    ├── evaluate.py     # 평가 스크립트
    └── utils.py        # 유틸리티 함수 (시각화, 체크포인트 저장 등)
```
