# Classification Modeling Summary

## 1. 역할과 모델링 목적

이번 단계의 담당 역할은 **Classification Modeling & Evaluation Lead**이다. 목표는 Kaggle Hotel Booking Demand dataset에서 예약 취소 여부를 나타내는 `is_canceled`를 예측하는 분류 모델을 만드는 것이다.

이 모델의 비즈니스 목적은 단순히 높은 정확도를 얻는 것이 아니라, 호텔이 예약 취소 가능성이 높은 고객을 사전에 파악하도록 돕는 것이다. 이를 통해 호텔은 다음과 같은 의사결정을 더 빨리 할 수 있다.

- 취소 위험이 높은 예약에 대해 사전 확인 연락 또는 프로모션 제공
- 초과 예약(overbooking) 전략 수립
- 객실, 인력, 식자재 등 운영 자원 배분
- 취소 가능성에 따른 수익 손실 관리

제안서에서 정의한 classification target은 `is_canceled`이고, 본 노트북도 이 target을 그대로 사용했다. 따라서 프로젝트의 핵심 목표인 **호텔 예약 취소 예측**과 논리적으로 일치한다.

관련 구현 파일은 `notebooks/03_classification_modeling.ipynb`이다.

## 2. 사용 데이터와 전처리 연결성

모델링에는 전처리 단계에서 생성한 `data/processed/hotel_bookings_clf.csv`를 사용했다.

데이터 기본 상태는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| 데이터셋 | `hotel_bookings_clf.csv` |
| 행 수 | 119,210 |
| 전체 컬럼 수 | 83 |
| 결측치 수 | 0 |
| object 컬럼 수 | 0 |
| target | `is_canceled` |
| 취소 안 됨 `0` | 75,011건, 62.92% |
| 취소됨 `1` | 44,199건, 37.08% |

전처리 단계와 이어지는 논리는 다음과 같다.

- 결측치는 전처리에서 처리되어 모델링 단계에서 별도 imputation이 필요하지 않다.
- 범주형 변수는 전처리에서 one-hot encoding되어 scikit-learn 모델에 바로 입력할 수 있다.
- `reservation_status`, `reservation_status_date`는 예약 결과 이후에 알 수 있는 컬럼이므로 전처리에서 제거되어 data leakage를 방지했다.
- `total_guests`, `total_stay`, `is_family`, `has_agent`, `has_company` 같은 파생 변수는 전처리에서 생성되어 모델의 입력 후보로 사용했다.
- scaling은 전처리에서 미리 하지 않고 모델링 단계의 pipeline 안에서 수행했다. 이는 train/test leakage를 막기 위한 전처리 요약의 방침과 일치한다.

## 3. Feature 선택 논리

Target 변수는 `is_canceled`이다. 따라서 모델 입력 `X`에서는 `is_canceled`를 반드시 제거했다.

추가로 다음 컬럼들을 입력에서 제외했다.

| 제외 대상 | 제외 이유 |
| --- | --- |
| `is_canceled` | 예측해야 하는 target이므로 입력에 포함하면 안 된다. |
| `adr` | 회귀 파트의 target이며, 제안서의 취소 예측 후보 변수에 직접 포함되지 않았다. 분류와 회귀 목표를 명확히 분리하기 위해 제외했다. |
| `assigned_room_type_*` | 실제 예약 시점에는 확정되지 않을 수 있는 정보이다. 전처리 요약에서도 prediction-time risk가 있다고 설명했기 때문에 보수적으로 제외했다. |

이 처리 후 최종 classification 입력 feature 수는 **71개**이다.

이 선택은 논리적으로 안전하다. `reserved_room_type_*`는 고객이 예약한 객실 타입이므로 예약 시점에 알 수 있지만, `assigned_room_type_*`는 호텔이 나중에 배정한 객실 타입일 수 있다. 취소 예측 모델은 “사전에” 취소 위험을 예측해야 하므로, 미래 또는 운영 이후 정보를 포함할 가능성이 있는 컬럼은 제거하는 것이 더 방어 가능한 선택이다.

전처리 요약에서는 `assigned_room_type`의 영향 확인을 위한 feature ablation 가능성을 언급했다. 본 classification baseline에서는 사전 예측이라는 비즈니스 목적을 우선하여 해당 컬럼을 제외했다. 이후 시간이 충분하면 같은 train/test split에서 `assigned_room_type_*`를 포함한 모델과 제외한 모델을 비교하는 추가 실험으로 확장할 수 있다.

## 4. 사용한 모델과 수업 범위

모델은 수업에서 다룬 scikit-learn 기반 supervised learning, model evaluation, ensemble learning 범위 안에서 구성했다.

사용한 모델은 다음 세 가지이다.

| 모델 | 사용 이유 |
| --- | --- |
| Logistic Regression | 단순하고 해석 가능한 baseline으로 사용했다. |
| Decision Tree | 비선형 조건과 규칙 기반 분류를 표현할 수 있다. |
| Random Forest | 여러 decision tree를 결합하는 ensemble learning 모델이다. |

Logistic Regression은 feature scale에 영향을 받을 수 있으므로 `StandardScaler`를 사용했다. 단, scaler를 전체 데이터에 미리 적용하지 않고 `Pipeline` 안에 넣었다. 이렇게 하면 cross validation의 각 train fold에서만 scaler가 학습되고 validation fold에는 transform만 적용된다. 따라서 scaling 과정에서도 data leakage가 발생하지 않는다.

Decision Tree와 Random Forest는 tree 기반 모델이므로 feature scale의 영향을 거의 받지 않는다. 그래서 별도 scaling 없이 사용했다.

## 5. 구현 절차

노트북의 전체 흐름은 다음과 같다.

1. `hotel_bookings_clf.csv` 로드
2. 결측치, object 컬럼, target 분포 확인
3. `is_canceled`, `adr`, `assigned_room_type_*` 제거 후 `X`, `y` 생성
4. `train_test_split`으로 80% train, 20% test 분리
5. `stratify=y`를 사용하여 train/test의 취소 비율 유지
6. Logistic Regression, Decision Tree, Random Forest 모델 정의
7. `StratifiedKFold(n_splits=5)`로 5-fold cross validation 수행
8. accuracy, precision, recall, F1-score 계산
9. F1-score가 가장 높은 모델 선택
10. 선택 모델을 train set 전체에 학습
11. test set에서 최종 성능 평가
12. confusion matrix와 주요 feature importance 확인

평가 기준은 accuracy만 사용하지 않았다. 취소 예측은 취소 고객을 놓치지 않는 것도 중요하고, 정상 예약을 지나치게 취소 위험으로 분류하지 않는 것도 중요하다. 따라서 precision과 recall의 균형을 보는 **F1-score**를 최종 모델 선택 기준으로 사용했다.

## 6. Cross Validation 결과

5-fold cross validation은 test set을 사용하지 않고 train set 내부에서만 수행했다. 따라서 test set은 최종 평가용으로 남겨두었다.

| 모델 | CV Accuracy | CV Precision | CV Recall | CV F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8401 | 0.8142 | 0.7371 | 0.7737 |
| Random Forest | 0.8477 | 0.8645 | 0.6989 | 0.7729 |
| Logistic Regression | 0.8075 | 0.8029 | 0.6373 | 0.7105 |

Random Forest는 accuracy와 precision이 가장 높았다. 그러나 recall이 0.6989로 Decision Tree보다 낮았다. 이 의미는 Random Forest가 “취소라고 예측한 경우”에는 더 정확하지만, 실제 취소 예약을 더 많이 놓칠 수 있다는 뜻이다.

Decision Tree는 F1-score가 0.7737로 가장 높고 recall도 0.7371로 가장 좋았다. 다만 Random Forest의 F1-score가 0.7729이므로 두 모델의 차이는 매우 작다. 따라서 “Decision Tree가 압도적으로 우수하다”가 아니라, **성능이 거의 비슷한 상황에서 Decision Tree가 recall과 F1-score가 약간 높고 모델 해석이 더 쉽기 때문에 최종 선택했다**고 설명하는 것이 정확하다.

## 7. 최종 Test Set 결과

최종 선택 모델은 **Decision Tree**이다.

| 모델 | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8399 | 0.8095 | 0.7431 | 0.7749 |

해석은 다음과 같다.

- Accuracy 0.8399: 전체 test 예약 중 약 84.0%를 올바르게 분류했다.
- Precision 0.8095: 모델이 “취소될 것”이라고 예측한 예약 중 약 81.0%가 실제로 취소되었다.
- Recall 0.7431: 실제 취소 예약 중 약 74.3%를 모델이 찾아냈다.
- F1-score 0.7749: precision과 recall의 균형이 비교 모델 중 가장 좋았다.

Confusion matrix는 다음과 같다.

| 실제 / 예측 | 예측: 취소 안 됨 | 예측: 취소됨 |
| --- | ---: | ---: |
| 실제 취소 안 됨 | 13,456 | 1,546 |
| 실제 취소됨 | 2,271 | 6,569 |

발표에서 이 matrix는 다음처럼 설명할 수 있다.

- 실제로 취소되지 않은 15,002건 중 13,456건을 정상 예약으로 맞췄다.
- 실제로 취소된 8,840건 중 6,569건을 취소 위험 예약으로 찾아냈다.
- 1,546건은 실제로 취소되지 않았지만 취소 위험으로 잘못 예측했다. 이는 호텔 입장에서는 추가 확인 연락 같은 비용으로 이어질 수 있다.
- 2,271건은 실제로 취소됐지만 모델이 놓친 예약이다. 이는 취소 리스크 관리에서 가장 줄이고 싶은 오류이다.

PPT와 최종 보고서에 바로 사용할 수 있도록 confusion matrix 그림은 다음 경로에 저장되도록 노트북을 보완했다.

- `reports/figures/confusion_matrix_decision_tree.png`

## 8. 주요 Feature 해석

최종 Decision Tree 모델에서 중요도가 높게 나온 상위 feature는 다음과 같다.

| 순위 | Feature | Importance |
| ---: | --- | ---: |
| 1 | `deposit_type_Non Refund` | 0.4300 |
| 2 | `market_segment_Online TA` | 0.1117 |
| 3 | `total_of_special_requests` | 0.0896 |
| 4 | `country_PRT` | 0.0759 |
| 5 | `lead_time` | 0.0750 |
| 6 | `required_car_parking_spaces` | 0.0413 |
| 7 | `previous_cancellations` | 0.0384 |
| 8 | `arrival_date_year` | 0.0351 |
| 9 | `booking_changes` | 0.0188 |
| 10 | `customer_type_Transient` | 0.0133 |

이 결과는 EDA 및 전처리 논리와 연결된다.

- EDA에서 취소율은 deposit type, market segment, customer type, lead time, special requests에 따라 차이가 있었다.
- 최종 모델에서도 `deposit_type_Non Refund`, `market_segment_Online TA`, `total_of_special_requests`, `lead_time`이 중요한 변수로 나타났다.
- `previous_cancellations`는 과거 취소 이력이 현재 예약 취소 가능성과 관련될 수 있다는 비즈니스 해석이 가능하다.
- `required_car_parking_spaces`와 `total_of_special_requests`는 고객의 예약 의도 또는 구체성이 반영된 변수로 볼 수 있다.

특히 `deposit_type_Non Refund`의 중요도가 가장 높게 나타났다. 이는 보증금 정책이 취소 행동과 강하게 관련되어 있음을 시사한다. 다만 feature importance는 인과관계를 의미하지 않는다. 즉, “Non Refund가 취소를 유발한다”가 아니라 “모델이 취소 여부를 구분할 때 이 변수를 많이 사용했다”로 해석해야 한다.

PPT와 최종 보고서에 바로 사용할 수 있도록 feature importance 그림은 다음 경로에 저장되도록 노트북을 보완했다.

- `reports/figures/classification_feature_importance.png`
- `reports/figures/classification_model_comparison.png`

## 9. 제안서와의 정합성 검토

제안서의 classification 계획과 현재 구현의 대응 관계는 다음과 같다.

| 제안서 요구 | 현재 구현 |
| --- | --- |
| Target은 `is_canceled` | 그대로 사용 |
| 예약 취소 여부 예측 | Decision Tree 기반 최종 모델 구현 |
| 후보 변수: `lead_time`, `deposit_type`, `customer_type`, `previous_cancellations`, `booking_changes`, `market_segment`, `required_car_parking_spaces`, `total_of_special_requests` 등 | 전처리된 one-hot/numeric feature로 포함 |
| categorical encoding | 전처리 단계에서 one-hot encoding 완료 |
| feature scaling | Logistic Regression에 대해 Pipeline 내부에서 적용 |
| data leakage prevention | `reservation_status`, `reservation_status_date` 제거 유지, `assigned_room_type_*` 추가 제외 |
| k-fold cross validation | `StratifiedKFold(n_splits=5)` 사용 |
| evaluation metrics | accuracy, precision, recall, F1-score, confusion matrix 사용 |

따라서 현재 classification modeling은 제안서의 문제 정의와 전처리 방향을 무너뜨리지 않는다. 오히려 `assigned_room_type_*`를 모델링 단계에서 제외하여 “사전 예측”이라는 비즈니스 목적에 더 맞게 보수적으로 조정했다.

## 10. 논리적으로 방어 가능한 지점

현재 구현에서 특히 방어 가능한 부분은 다음과 같다.

- **Target leakage 방지**: `is_canceled`는 입력에서 제거했고, 전처리에서 이미 `reservation_status`, `reservation_status_date`를 제거했다.
- **Task 분리**: `adr`는 regression target이므로 classification input에서 제외했다. 이로 인해 취소 예측과 ADR 예측의 목적이 섞이지 않는다.
- **Prediction-time risk 관리**: `assigned_room_type_*`는 예약 이후 배정될 가능성이 있으므로 제외했다.
- **분포 유지**: `stratify=y`로 train/test의 취소 비율을 유지했다.
- **Cross validation**: 단일 train/test split에만 의존하지 않고 5-fold CV로 모델 안정성을 비교했다.
- **Scaling leakage 방지**: scaling은 pipeline 안에서 수행되어 train fold 기준으로만 학습된다.
- **평가 지표 균형**: accuracy만 보지 않고 precision, recall, F1-score를 함께 사용했다.

## 11. 발표용 설명 포인트

발표에서는 다음 흐름으로 설명하면 자연스럽다.

1. 우리 팀의 classification 목표는 `is_canceled` 예측이다.
2. 전처리된 classification dataset은 119,210행, 83컬럼이며 결측치와 문자열 컬럼이 없다.
3. 입력에서는 target, 회귀 target인 `adr`, 예측 시점 위험이 있는 `assigned_room_type_*`를 제외했다.
4. train/test split은 80:20으로 했고, 취소 비율 유지를 위해 stratified split을 사용했다.
5. Logistic Regression, Decision Tree, Random Forest를 비교했다.
6. 평가는 5-fold cross validation으로 수행했고, accuracy, precision, recall, F1-score를 계산했다.
7. 최종 선택 기준은 F1-score이다. 취소 예측은 precision과 recall의 균형이 중요하기 때문이다.
8. Decision Tree와 Random Forest의 성능은 거의 비슷했지만, Decision Tree가 CV F1 0.7737로 가장 높고 recall도 더 높아 최종 모델이 되었다.
9. Test set에서도 accuracy 0.8399, precision 0.8095, recall 0.7431, F1 0.7749로 안정적인 결과를 보였다.
10. Confusion matrix 기준 실제 취소 8,840건 중 6,569건을 찾아냈다.
11. 중요한 변수는 `deposit_type_Non Refund`, `market_segment_Online TA`, `total_of_special_requests`, `country_PRT`, `lead_time` 등이었다.
12. 이 변수들은 EDA에서 확인한 취소 패턴과 연결되므로 모델 결과 해석도 자연스럽다.

## 12. 한계와 다음 단계

현재 모델은 모델링 파트의 baseline으로 충분히 사용할 수 있다. 다만 다음 개선은 선택적으로 고려할 수 있다.

- Decision Tree의 `max_depth` 값을 더 다양하게 바꾸며 overfitting 여부를 비교할 수 있다.
- Random Forest는 precision이 높았으므로, 호텔이 false alarm을 줄이고 싶다면 Random Forest도 후보로 유지할 수 있다.
- `assigned_room_type_*` 포함/제외 feature ablation을 추가하면 전처리 요약에서 언급한 prediction-time risk를 실험적으로 더 명확히 설명할 수 있다.
- 취소 예약 recall을 더 높이는 것이 목적이면 class weight 또는 SMOTE를 검토할 수 있다. 단, 현재 target 비율은 62.92% 대 37.08%로 극단적인 imbalance는 아니므로 이번 baseline에서는 사용하지 않았다.
- 최종 보고서에서는 confusion matrix를 함께 제시해 단순 accuracy보다 비즈니스 오류 비용을 중심으로 해석하는 것이 좋다.
- KNN은 수업 범위에 포함될 수 있지만, 전체 데이터 119,210행에 대해 5-fold cross validation을 수행하면 거리 계산 비용이 크다. 따라서 이번 baseline에서는 Logistic Regression, Decision Tree, Random Forest로 모델 비교를 구성했고, 필요하면 이후 작은 stratified sample에서 KNN을 보조 실험으로 추가할 수 있다.

## 13. 결론

현재 classification modeling은 제안서의 목표, EDA에서 확인한 변수 관계, 전처리 단계의 leakage prevention 논리와 일관된다. 최종 모델은 Decision Tree이며, 5-fold cross validation과 hold-out test 평가에서 모두 안정적인 F1-score를 보였다.

따라서 현재 결과는 **분류 모델링 파트의 1차 완성본으로 사용 가능**하다. 발표에서는 “Decision Tree와 Random Forest의 성능 차이는 매우 작지만, Decision Tree가 recall과 F1-score가 약간 높고 해석 가능성이 좋아 최종 선택했다”는 논리로 설명하면 된다.
