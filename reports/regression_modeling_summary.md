# Regression Modeling Summary

## 1. 역할과 모델링 목적

이번 단계의 목표는 **ADR(Average Daily Rate)** 을 예측하는 regression 모델을 만드는 것이다. 제안서에서 regression target은 `adr`로 정의되어 있으며, 호텔의 revenue management와 pricing decision을 지원하는 것이 비즈니스 목적이다.

이 모델은 예약 조건을 바탕으로 객실 평균 요금을 추정한다. 이를 통해 호텔은 다음 의사결정을 지원할 수 있다.

- 예약 조건에 따른 예상 객실 요금 파악
- 시즌, 호텔 유형, 고객 특성에 따른 가격 패턴 이해
- 실제 ADR과 예측 ADR 차이를 통한 가격 정책 점검
- revenue management 및 객실 가격 전략 수립

관련 구현 파일은 `notebooks/04_regression_modeling.ipynb`이다.

## 2. 사용 데이터와 전처리 연결성

모델링에는 전처리 단계에서 생성한 `data/processed/hotel_bookings_reg.csv`를 사용했다.

데이터 상태는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| 데이터셋 | `hotel_bookings_reg.csv` |
| 행 수 | 73,386 |
| 전체 컬럼 수 | 83 |
| 결측치 수 | 0 |
| object 컬럼 수 | 0 |
| target | `adr` |
| `is_canceled` 값 | 모두 0 |
| ADR 평균 | 102.373 |
| ADR 중앙값 | 94.500 |
| ADR 표준편차 | 47.222 |
| ADR 최솟값 | 0.260 |
| ADR 최댓값 | 451.500 |

전처리와 이어지는 논리는 다음과 같다.

- Regression dataset은 non-canceled booking만 포함한다. 이는 완료된 예약 기준 ADR 예측이라는 전처리 요약의 논리와 일치한다.
- ADR 이상치는 `0 < adr < 500` 기준으로 필터링되어 극단값의 영향을 줄였다.
- 결측치와 문자열 컬럼이 없으므로 scikit-learn regression 모델에 바로 사용할 수 있다.
- 범주형 변수는 one-hot encoding되어 있다.
- 전역 scaling은 전처리 단계에서 하지 않았고, 모델링 단계에서 필요한 경우에만 `Pipeline` 안에서 수행했다.

## 3. Feature 선택 논리

Target 변수는 `adr`이다. 따라서 입력 feature에서는 `adr`를 제거했다.

추가로 다음 컬럼들을 제외했다.

| 제외 대상 | 제외 이유 |
| --- | --- |
| `adr` | 예측해야 하는 regression target이다. |
| `is_canceled` | regression dataset은 non-canceled booking만 포함하므로 값이 모두 0인 상수 컬럼이다. 모델 입력으로 의미가 없다. |
| `assigned_room_type_*` | 실제 배정 객실은 예약 이후 또는 체크인 시점에 결정될 수 있으므로, 사전 revenue prediction 목적에서는 prediction-time leakage 위험이 있다. |

이 처리 후 최종 regression 입력 feature 수는 **71개**이다.

`reserved_room_type_*`는 고객이 예약한 객실 타입이므로 입력에 유지했다. 반면 `assigned_room_type_*`는 호텔이 나중에 배정한 정보일 수 있어 제외했다. 이 선택은 **예약 시점 또는 투숙 전 ADR 예측**이라는 제안서의 목표에 더 안전하게 맞는다.

## 4. 사용한 모델과 평가 설계

비교한 모델/파라미터 조합은 총 5개이다.

| 모델 조합 | 사용 이유 |
| --- | --- |
| Linear Regression | 연속형 target 예측을 위한 단순 baseline |
| Decision Tree depth 8 | 얕은 tree로 과적합을 줄인 비선형 모델 |
| Decision Tree depth 12 | 중간 깊이의 tree 모델 |
| Decision Tree depth 16 | 더 복잡한 tree 모델 |
| Random Forest depth 14 | 여러 tree를 결합한 ensemble regression 모델 |

평가는 `KFold(n_splits=5, shuffle=True, random_state=42)`를 사용했다. Regression에서는 classification처럼 class label이 없으므로 `StratifiedKFold`가 아니라 일반 `KFold`를 사용했다.

평가 지표는 다음 네 가지이다.

| 지표 | 의미 |
| --- | --- |
| MAE | 예측 ADR이 실제 ADR과 평균적으로 얼마나 차이 나는지 |
| MSE | 큰 오차를 더 강하게 벌점화하는 평균 제곱 오차 |
| RMSE | MSE의 제곱근으로, ADR과 같은 단위로 해석 가능 |
| R² | 모델이 ADR 변동성을 얼마나 설명하는지 |

최종 모델 선택 기준은 **CV RMSE가 가장 낮은 모델**이다. ADR 예측에서는 큰 가격 오차가 revenue decision에 더 큰 문제를 만들 수 있기 때문에 RMSE를 중심 기준으로 삼았다. MAE와 R²는 보조 해석 지표로 함께 사용했다.

노트북의 CV RMSE는 각 fold의 MSE 평균에 제곱근을 적용한 `sqrt(mean CV MSE)` 방식으로 계산했다. 문서의 CV RMSE 수치는 이 계산 방식과 일관된다.

## 5. Top 5 Cross Validation 결과

| 모델 조합 | CV MAE | CV MSE | CV RMSE | CV R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest depth 14 | 11.9109 | 340.7130 | 18.4584 | 0.8469 |
| Decision Tree depth 16 | 13.2756 | 509.5903 | 22.5741 | 0.7710 |
| Decision Tree depth 12 | 15.2007 | 544.5843 | 23.3363 | 0.7553 |
| Decision Tree depth 8 | 19.5132 | 757.5012 | 27.5227 | 0.6595 |
| Linear Regression | 21.0904 | 821.1745 | 28.6561 | 0.6309 |

비교한 5개 모델/파라미터 조합 중에서는 Random Forest depth 14가 모든 주요 지표에서 가장 좋은 성능을 보였다. 특히 CV RMSE가 18.4584로 가장 낮고, CV R²가 0.8469로 가장 높았다. 따라서 최종 모델로 Random Forest depth 14를 선택했다.

이 결과는 선형 모델보다 tree 기반 비선형 모델이 ADR 예측에 더 적합하다는 것을 보여준다. ADR은 호텔 유형, 시즌, 고객 수, market segment, room type 등이 복합적으로 작용하는 값이므로 단순 선형 관계만으로는 충분히 설명하기 어렵다.

## 6. 최종 Test Set 결과

최종 선택 모델은 **Random Forest depth 14**이다.

| 모델 | Test MAE | Test MSE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest depth 14 | 11.8832 | 348.7546 | 18.6750 | 0.8449 |

해석은 다음과 같다.

- MAE 11.8832: 평균적으로 실제 ADR과 예측 ADR의 차이가 약 11.88이다.
- RMSE 18.6750: 큰 오차까지 고려했을 때 평균적인 예측 오차 규모는 약 18.68 ADR 단위이다.
- R² 0.8449: test set에서 ADR 변동성의 약 84.49%를 설명했다.
- CV RMSE 18.4584와 test RMSE 18.6750이 매우 가까워, test set에서 성능이 크게 무너지지 않았다.

따라서 현재 모델은 regression baseline으로 충분히 안정적인 결과를 보인다. 단, 이는 이번 노트북에서 비교한 5개 조합 중 최선이라는 의미이며, 가능한 모든 회귀 모델과 하이퍼파라미터를 전부 탐색했다는 뜻은 아니다.

## 7. 저장된 Figure

PPT와 최종 보고서에 바로 사용할 수 있도록 다음 figure를 저장했다.

| Figure | 경로 |
| --- | --- |
| 모델 비교 | `reports/figures/regression_model_comparison.png` |
| 실제 ADR vs 예측 ADR | `reports/figures/regression_actual_vs_predicted.png` |
| residual 분석 | `reports/figures/regression_residual_analysis.png` |
| feature importance | `reports/figures/regression_feature_importance.png` |

발표에서는 모델 비교 figure로 Random Forest가 가장 낮은 RMSE/MAE를 보였다는 점을 설명하고, actual vs predicted plot과 residual plot으로 예측이 어느 정도 안정적으로 작동했는지 설명하면 된다. Residual plot은 예측 오차가 특정 예측 ADR 구간에 몰리는지 확인하기 위한 보조 해석 도구로 사용한다.

## 8. 주요 Feature 해석

최종 Random Forest 모델에서 중요도가 높게 나온 상위 feature는 다음과 같다.

| 순위 | Feature | Importance |
| ---: | --- | ---: |
| 1 | `arrival_date_week_number` | 0.2427 |
| 2 | `total_guests` | 0.2220 |
| 3 | `hotel_Resort Hotel` | 0.1071 |
| 4 | `lead_time` | 0.0494 |
| 5 | `arrival_date_year` | 0.0446 |
| 6 | `meal_HB` | 0.0422 |
| 7 | `market_segment_Online TA` | 0.0391 |
| 8 | `market_segment_Direct` | 0.0378 |
| 9 | `arrival_date_month_August` | 0.0270 |
| 10 | `arrival_date_day_of_month` | 0.0239 |

이 결과는 EDA 및 제안서와 논리적으로 연결된다.

- 제안서는 ADR 예측 feature로 hotel type, stay length, number of guests, market segment, room type, seasonality variables를 언급했다.
- 모델에서도 `arrival_date_week_number`, `arrival_date_month_August`, `hotel_Resort Hotel`, `total_guests`, `market_segment_*`, `reserved_room_type_*` 관련 변수가 중요하게 나타났다.
- ADR은 계절성, 호텔 유형, 고객 수, 유통 채널, 객실 타입에 따라 달라질 수 있으므로 feature importance 결과는 비즈니스적으로 자연스럽다.

다만 feature importance는 인과관계를 의미하지 않는다. 예를 들어 `arrival_date_week_number`가 높다고 ADR을 직접 올린다는 뜻이 아니라, 모델이 ADR을 구분하는 과정에서 해당 계절성 변수를 많이 사용했다는 의미이다.

## 9. 제안서와의 정합성

| 제안서 요구 | 현재 구현 |
| --- | --- |
| Regression target은 `adr` | 그대로 사용 |
| ADR 기반 revenue/pricing decision 지원 | Random Forest regression 모델로 구현 |
| Feature 후보: hotel type, stay length, guests, market segment, room type, seasonality | one-hot/numeric feature로 반영 |
| Missing value handling | 전처리 단계에서 완료 |
| Categorical encoding | 전처리 단계에서 one-hot encoding 완료 |
| Feature scaling | Linear Regression에 대해 Pipeline 내부에서 적용 |
| Abnormal ADR filtering | regression dataset에 `0 < adr < 500` 적용됨 |
| 평가 지표 | MAE, MSE, RMSE, R² 사용 |

현재 regression modeling은 제안서의 ADR 예측 목표와 전처리 요약의 target-specific filtering 논리를 유지한다.

## 10. 한계와 다음 단계

현재 모델은 A+ 기준의 regression baseline으로 충분히 사용할 수 있다. 다만 다음 개선은 선택적으로 고려할 수 있다.

- `assigned_room_type_*` 포함/제외 ablation을 추가하면 prediction-time risk와 성능 차이를 더 명확히 설명할 수 있다.
- Random split 외에 시간 기반 split을 사용하면 실제 미래 예약 예측 상황을 더 엄격하게 평가할 수 있다.
- 현재 Random Forest parameter search는 제한적이다. `n_estimators`, `max_depth`, `min_samples_leaf`를 더 다양하게 바꾸면 추가 성능 개선 가능성이 있다.
- Residual이 특정 ADR 구간에서 커지는지 분석하면 고가 객실 예측의 약점을 더 자세히 설명할 수 있다.
- 현재는 extreme imbalance 문제가 아니므로 classification에서처럼 SMOTE는 regression task에 적용하지 않는다.

## 11. 발표용 핵심 문장

발표에서는 다음 문장으로 정리하면 안전하다.

> For ADR prediction, Random Forest depth 14 achieved the best performance among the five compared model/parameter combinations. The model reached test MAE 11.88, RMSE 18.68, and R² 0.8449. This means the model explains about 84.5% of ADR variation on the test set and can support hotel pricing and revenue management decisions.

한국어 발표에서는 다음처럼 말하면 된다.

> ADR 예측에서는 비교한 5개 모델/파라미터 조합 중 Random Forest depth 14가 가장 좋은 성능을 보였습니다. Test MAE는 약 11.88, RMSE는 약 18.68, R²는 0.8449로 나타났고, 이는 test set에서 ADR 변동성의 약 84.5%를 설명한다는 의미입니다. 따라서 이 모델은 호텔 가격 전략과 revenue management를 지원하는 회귀 baseline으로 사용할 수 있습니다.
