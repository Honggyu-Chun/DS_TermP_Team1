# Regression Modeling User Manual

## 1. 목적

이 문서는 `src/regression.py`의 top-level function인 `run_regression_modeling()` 사용 방법을 설명한다. 과제 요구사항의 Open Source SW 항목에서는 전처리, 모델 학습, 평가를 반복 코드가 아니라 하나의 상위 함수 형태로 정리하고, 함수 설명서와 사용법을 제공할 것을 요구한다.

`run_regression_modeling()`은 전처리된 호텔 예약 regression dataset을 입력받아 ADR을 예측하는 모델을 학습하고 평가한다.

## 2. 함수 위치

```text
src/regression.py
```

Top-level function:

```python
run_regression_modeling(
    data_path="data/processed/hotel_bookings_reg.csv",
    output_dir="reports/figures",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_figures=True,
)
```

## 3. 입력 데이터

기본 입력 파일은 다음과 같다.

```text
data/processed/hotel_bookings_reg.csv
```

이 데이터는 전처리 단계에서 생성된 회귀용 데이터셋이다.

| 항목 | 내용 |
| --- | --- |
| Target | `adr` |
| 데이터 범위 | non-canceled booking |
| 결측치 | 없음 |
| 문자열 컬럼 | 없음 |
| 인코딩 | 범주형 변수 one-hot encoding 완료 |
| ADR 필터링 | `0 < adr < 500` |

## 4. 파라미터 설명

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `data_path` | `data/processed/hotel_bookings_reg.csv` | 회귀 모델링에 사용할 CSV 경로 |
| `output_dir` | `reports/figures` | 모델링 결과 figure를 저장할 폴더 |
| `test_size` | `0.2` | train/test split에서 test set 비율 |
| `n_splits` | `5` | K-fold cross validation fold 수 |
| `random_state` | `42` | 재현 가능한 split과 모델 학습을 위한 random seed |
| `save_figures` | `True` | 결과 figure 저장 여부 |

## 5. 반환값

함수는 dictionary를 반환한다.

| Key | 설명 |
| --- | --- |
| `cv_result` | 모든 모델/파라미터 조합의 cross validation 결과 |
| `top5_result` | CV RMSE 기준 상위 5개 조합 |
| `test_result` | 최종 선택 모델의 test set 평가 결과 |
| `feature_importance` | 최종 모델의 feature importance |
| `best_model_name` | CV RMSE 기준 최종 선택 모델명 |
| `best_model` | 학습 완료된 최종 모델 객체 |
| `feature_columns` | 모델 입력 feature 목록 |
| `dropped_columns` | 모델 입력에서 제외한 컬럼 목록 |

## 6. 사용 예시

프로젝트 루트에서 다음처럼 실행할 수 있다.

```python
from src.regression import run_regression_modeling

result = run_regression_modeling()

print(result["best_model_name"])
print(result["top5_result"].round(4))
print(result["test_result"].round(4))
```

터미널에서는 다음 명령으로도 실행할 수 있다.

```bash
python src/regression.py
```

## 7. 내부 처리 흐름

함수는 다음 순서로 동작한다.

1. 전처리된 regression CSV를 불러온다.
2. `adr`를 target으로 설정한다.
3. 입력 feature에서 `adr`, `is_canceled`, `assigned_room_type_*`를 제거한다.
4. 데이터를 train/test set으로 나눈다.
5. 5개 모델/파라미터 조합을 5-fold cross validation으로 비교한다.
6. CV RMSE가 가장 낮은 모델을 최종 모델로 선택한다.
7. 최종 모델을 test set에서 MAE, MSE, RMSE, R²로 평가한다.
8. 모델 비교, 실제값 vs 예측값, residual 분석, feature importance figure를 저장한다.

## 8. 비교 모델과 수업 범위

함수에서 사용하는 모델은 강의 범위 안의 기본적인 supervised learning 모델로 제한했다.

| 모델 조합 | 사용 이유 |
| --- | --- |
| Linear Regression | 회귀 baseline |
| Decision Tree depth 8 | 얕은 tree 기반 비선형 모델 |
| Decision Tree depth 12 | 중간 깊이 tree |
| Decision Tree depth 16 | 더 복잡한 tree |
| Random Forest depth 14 | ensemble learning 기반 회귀 모델 |

Linear Regression은 feature scale의 영향을 받을 수 있으므로 `Pipeline` 안에서 `StandardScaler`를 적용했다. 이는 train/test split 이후 학습 fold 안에서 scaling을 적용하므로 data leakage를 줄이는 방식이다.

## 9. 저장되는 출력 파일

`save_figures=True`일 때 다음 figure가 저장된다.

| 파일 | 설명 |
| --- | --- |
| `regression_model_comparison.png` | CV 기준 모델 성능 비교 |
| `regression_actual_vs_predicted.png` | 실제 ADR과 예측 ADR 비교 |
| `regression_residual_analysis.png` | residual 분포와 residual pattern 확인 |
| `regression_feature_importance.png` | 최종 모델의 주요 feature |

## 10. 현재 결과 요약

기본 설정으로 실행한 결과, 비교한 5개 조합 중 **Random Forest depth 14**가 가장 좋은 성능을 보였다.

| 지표 | 값 |
| --- | ---: |
| Test MAE | 11.8832 |
| Test MSE | 348.7546 |
| Test RMSE | 18.6750 |
| Test R² | 0.8449 |

이 결과는 test set에서 ADR 변동성의 약 84.49%를 설명한다는 의미이다. 따라서 호텔의 가격 전략과 revenue management를 지원하는 회귀 baseline으로 사용할 수 있다.

## 11. 논리적 주의사항

`assigned_room_type_*`는 성능을 높일 수 있는 변수일 수 있지만, 예약 시점 또는 투숙 전 예측 상황에서는 아직 확정되지 않았을 수 있다. 따라서 현재 함수는 더 보수적인 prediction-time 기준을 적용하여 해당 변수를 제외한다.

Feature importance는 인과관계를 의미하지 않는다. 이는 모델이 예측 과정에서 어떤 변수를 많이 사용했는지 보여주는 해석 도구이다.
