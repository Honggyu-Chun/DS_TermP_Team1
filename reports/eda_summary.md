\# EDA Summary



\## 데이터 크기

\- 행: 119,390 / 열: 32





\## 변수 종류

\- 수치형 변수: 20개 (lead\_time, adr, adults 등)

\- 범주형 변수: 12개 (hotel, meal, country 등)

\- 이진 변수: hotel, is\_canceled, is\_repeated\_guest





\## 결측치

| 변수 | 결측치 수 | 비율 |

|------|--------|------|

| children | 4 | 0.003% |

| country | 488 | 0.41% |

| agent | 16,340 | 13.69% |

| company | 112,593 | 94.31% |





\## 기초 통계량

\- 취소율: 37.04%

\- ADR 평균: 101.83 / 최대값: 5,400

\- lead\_time 평균: 104일 / 최대값: 737일





\## 이상치

\- ADR 최대값 5,400으로 이상치 존재

\- ADR 0 이하인 행: 1,960개

\- total guests가 0인 행: 180개

