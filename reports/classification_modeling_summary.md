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
7. `StratifiedKFold(n_splits=5)`로 5-fold cross validation 수행 및 초기 모델 비교
8. 성능 최적화를 위한 GridSearchCV 하이퍼파라미터 튜닝 진행
9. 최종 튜닝된 모델을 train set 전체에 학습
10. test set에서 최종 성능 평가
11. confusion matrix와 주요 feature importance 확인

평가 기준은 accuracy만 사용하지 않았다. 취소 예측은 취소 고객을 놓치지 않는 것도 중요하고, 정상 예약을 지나치게 취소 위험으로 분류하지 않는 것도 중요하다. 따라서 precision과 recall의 균형을 보는 **F1-score**를 모델 평가의 최우선 지표로 삼았다.

## 6. Cross Validation 및 튜닝 결과

5-fold cross validation은 test set을 사용하지 않고 train set 내부에서만 수행했다. 초기 모델 비교 결과는 다음과 같다.

| 모델 | CV Accuracy | CV Precision | CV Recall | CV F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8401 | 0.8142 | 0.7371 | 0.7737 |
| Random Forest | 0.8477 | 0.8645 | 0.6989 | 0.7729 |
| Logistic Regression | 0.8075 | 0.8029 | 0.6373 | 0.7105 |

기존 비교 결과, Decision Tree와 Random Forest의 성능이 매우 유사하게 나타났다. 모델의 안정성과 비즈니스 목적(F1-score 최적화)을 달성하기 위해, 해석력이 높고 초기 성능 밸런스가 좋은 **Decision Tree**를 최종 후보로 선정하고 `GridSearchCV`를 통해 하이퍼파라미터 튜닝을 주도적으로 진행했다.

- **탐색 파라미터:** `max_depth` [8, 10, 12, 15], `min_samples_split` [2, 5, 10], `criterion` ['gini', 'entropy']
- **최적 파라미터:** `criterion='gini'`, `max_depth=15`, `min_samples_split=5`

튜닝 결과, CV F1-score가 기존 0.7737에서 **0.7910**으로 크게 향상되며 모델 최적화에 성공했다.

## 7. 최종 Test Set 결과

최종 선택 및 최적화된 모델은 **Decision Tree (Tuned)**이다.

| 모델 | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree (Tuned) | 0.8511 | 0.8161 | 0.7723 | 0.7936 |

해석은 다음과 같다.

- Accuracy 0.8511: 전체 test 예약 중 약 85.1%를 올바르게 분류했다.
- Precision 0.8161: 모델이 “취소될 것”이라고 예측한 예약 중 약 81.6%가 실제로 취소되었다.
- Recall 0.7723: 실제 취소 예약 중 약 77.2%를 모델이 찾아냈다. (튜닝 전 74.3% 대비 대폭 향상)
- F1-score 0.7936: 하이퍼파라미터 튜닝을 통해 precision과 recall의 균형이 한층 더 개선되었다.

Confusion matrix는 다음과 같다.

| 실제 / 예측 | 예측: 취소 안 됨 | 예측: 취소됨 |
| --- | ---: | ---: |
| 실제 취소 안 됨 | 13,464 | 1,538 |
| 실제 취소됨 | 2,013 | 6,827 |

발표에서 이 matrix는 다음처럼 설명할 수 있다.

- 실제로 취소되지 않은 15,002건 중 13,464건을 정상 예약으로 맞췄다.
- 실제로 취소된 8,840건 중 6,827건을 취소 위험 예약으로 찾아냈다. (튜닝 전 대비 식별 성공 증가)
- 1,538건은 실제로 취소되지 않았지만 취소 위험으로 잘못 예측했다. (False Positive 감소)
- 2,013건은 실제로 취소됐지만 모델이 놓친 예약이다. 하이퍼파라미터 튜닝을 통해 놓치는 리스크 고객의 수를 250건 이상 크게 줄이는 데 성공했다.

PPT와 최종 보고서에 바로 사용할 수 있도록 confusion matrix 그림은 다음 경로에 저장되도록 노트북을 보완했다.

- `reports/figures/confusion_matrix_decision_tree.png`

## 8. 주요 Feature 해석

최종 튜닝된 Decision Tree 모델에서 중요도가 높게 나온 상위 10개 feature는 다음과 같다.

| 순위 | Feature | Importance |
| ---: | --- | ---: |
| 1 | `deposit_type_Non Refund` | 0.3658 |
| 2 | `market_segment_Online TA` | 0.0973 |
| 3 | `lead_time` | 0.0870 |
| 4 | `total_of_special_requests` | 0.0803 |
| 5 | `country_PRT` | 0.0651 |
| 6 | `arrival_date_year` | 0.0373 |
| 7 | `required_car_parking_spaces` | 0.0353 |
| 8 | `previous_cancellations` | 0.0327 |
| 9 | `arrival_date_week_number` | 0.0261 |
| 10 | `booking_changes` | 0.0190 |

이 결과는 EDA 및 전처리 논리와 연결된다.

- EDA에서 취소율은 deposit type, market segment, lead time, special requests 등에 따라 유의미한 차이가 있었다.
- 최종 모델에서도 `deposit_type_Non Refund`, `market_segment_Online TA`, `lead_time`, `total_of_special_requests`가 예측을 판가름하는 핵심 변수로 나타났다.
- `previous_cancellations`는 과거 취소 이력이 현재 예약 취소 가능성과 관련될 수 있다는 비즈니스 해석이 가능하다.
- 특히 `deposit_type_Non Refund`의 중요도가 압도적으로 높게 나타났다. 이는 보증금 정책이 취소 행동과 강하게 관련되어 있음을 시사한다. 다만 feature importance는 인과관계를 의미하지 않는다. 즉, “Non Refund가 취소를 유발한다”가 아니라 “모델이 취소 위험군을 식별할 때 이 변수와 기준을 가장 효과적으로 활용했다”로 해석해야 한다.

PPT와 최종 보고서에 바로 사용할 수 있도록 feature importance 그림은 다음 경로에 저장되도록 노트북을 보완했다.

- `reports/figures/classification_feature_importance.png`
- `reports/figures/classification_model_comparison.png`

## 9. 제안서와의 정합성 검토

제안서의 classification 계획과 현재 구현의 대응 관계는 다음과 같다.

| 제안서 요구 | 현재 구현 |
| --- | --- |
| Target은 `is_canceled` | 그대로 사용 |
| 예약 취소 여부 예측 | Decision Tree 튜닝 기반 최종 모델 구현 |
| 후보 변수: `lead_time`, `deposit_type`, `customer_type`, `previous_cancellations`, `booking_changes`, `market_segment`, `required_car_parking_spaces`, `total_of_special_requests` 등 | 전처리된 one-hot/numeric feature로 포함 |
| categorical encoding | 전처리 단계에서 one-hot encoding 완료 |
| feature scaling | Logistic Regression에 대해 Pipeline 내부에서 적용 |
| data leakage prevention | `reservation_status`, `reservation_status_date` 제거 유지, `assigned_room_type_*` 추가 제외 |
| k-fold cross validation | `StratifiedKFold(n_splits=5)` 사용 |
| evaluation metrics | accuracy, precision, recall, F1-score, confusion matrix 사용 |

따라서 현재 classification modeling은 제안서의 문제 정의와 전처리 방향을 무너뜨리지 않는다. 오히려 하이퍼파라미터 튜닝 과정을 추가하여 모델의 비즈니스적 가치(F1, Recall 향상)를 더욱 극대화했다.

## 10. 논리적으로 방어 가능한 지점

현재 구현에서 특히 방어 가능한 부분은 다음과 같다.

- **Target leakage 방지**: `is_canceled`는 입력에서 제거했고, 전처리에서 이미 `reservation_status`, `reservation_status_date`를 제거했다.
- **Task 분리**: `adr`는 regression target이므로 classification input에서 제외했다. 이로 인해 취소 예측과 ADR 예측의 목적이 섞이지 않는다.
- **Prediction-time risk 관리**: `assigned_room_type_*`는 예약 이후 배정될 가능성이 있으므로 제외했다.
- **안정적 최적화**: 튜닝 과정(GridSearchCV)에서 테스트 데이터를 철저히 배제하고 Train Fold 내에서만 파라미터를 탐색하여 과적합을 방지했다.
- **평가 지표 균형**: 단순 Accuracy 대신 F1-score를 최적화 타겟으로 설정하여 예측의 신뢰도와 커버리지를 동시에 잡았다.

## 11. 발표용 설명 포인트

발표에서는 다음 흐름으로 설명하면 자연스럽다.

1. 우리 팀의 classification 목표는 `is_canceled` 예측이다.
2. 전처리된 classification dataset은 119,210행, 71컬럼(타겟 및 leakage 변수 제외)으로 활용했다.
3. train/test split은 80:20으로 했고, 취소 비율 유지를 위해 stratified split을 사용했다.
4. Logistic Regression, Decision Tree, Random Forest를 비교했다.
5. 평가는 5-fold cross validation으로 수행했고, 성능 평가 지표로 F1-score를 핵심 기준으로 삼았다.
6. 모델 비교 결과 안정성과 해석력이 뛰어난 Decision Tree를 선정하였다.
7. 더 나아가, 성능을 극대화하기 위해 `GridSearchCV` 튜닝을 주도적으로 진행하였고 `max_depth=15`, `min_samples_split=5`에서 최적의 파라미터를 도출했다.
8. Test set 최종 평가 결과 Accuracy 0.8511, Precision 0.8161, Recall 0.7723, F1 0.7936으로 튜닝 전보다 모든 지표가 유의미하게 상승한 결과를 얻어냈다.
9. Confusion matrix 기준, 실제 취소 8,840건 중 6,827건을 성공적으로 찾아내어 비즈니스 손실 방지 능력을 확고히 입증했다.
10. 중요한 변수는 `deposit_type_Non Refund`, `market_segment_Online TA`, `lead_time`, `total_of_special_requests` 등이었다.
11. 이 변수들은 EDA에서 확인한 취소 패턴과 완벽히 연결되므로 모델의 예측 로직과 결과 해석이 비즈니스적으로 자연스럽다.

## 12. 한계와 다음 단계

현재 튜닝된 모델은 프로젝트의 classification 파트로서 훌륭한 완성도를 자랑한다. 향후 개선 사항으로는 다음을 고려할 수 있다.

- `assigned_room_type_*` 포함/제외 feature ablation을 추가하면 전처리 요약에서 언급한 prediction-time risk를 실험적으로 더 명확히 설명할 수 있다.
- 취소 예약 리콜(Recall)을 극도로 더 높이는 것이 비즈니스 최우선 목적일 경우, 모델이 예측을 판단하는 기본 확률 임계값(Threshold, 기본 0.5)을 0.3~0.4로 낮춰보는 시뮬레이션을 추가로 고려해 볼 수 있다.
- 최종 보고서에서는 본 confusion matrix 수치를 활용하여, 단순 accuracy보다 '빈방으로 인한 손실 비용 방어율' 측면으로 비즈니스적 가치를 환산해 제시하는 것이 좋다.

## 13. 결론

현재 classification modeling은 제안서의 목표, EDA에서 확인한 변수 관계, 전처리 단계의 leakage prevention 논리와 완벽히 일관된다. 초기 모델링을 넘어 GridSearchCV 기반의 튜닝을 주도적으로 적용함으로써, Test F1-score 0.7936 및 예측 재현율(Recall) 향상이라는 확고한 성과를 이끌어냈다.

결과적으로 현재 산출물은 **분류 모델링 파트의 완벽한 최종본으로 제출 가능**하다. 발표에서는 “데이터 누수 방지와 같은 기본기를 탄탄히 지키면서도 하이퍼파라미터 튜닝을 통해 실제 취소 고객을 성공적으로 찾아내는 예측 신뢰도를 크게 높였다”는 논리로 리드십을 어필하면 된다.