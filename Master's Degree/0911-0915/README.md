# 9월 11일 ~ 15일 (9월 3주차) 실험내용

1. Keydates에서 Risk-Index로 기준자료를 변환하여서 실험한 결과
- ConfusionMatrix를 직접 print해본 결과, label간의 데이터 Imbalance가 존재함을 확인함
- 실험 결과도 Data Imbalance에 영향을 받아서, 예측결과가 label '1'로 다 예측함. >> label '0'과 '2'를 제대로 예측해내지 못함

2. Classifier을 변경해가면서 실험해봄.
- BalancedRandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced', random_state=0, sampling_strategy = 'all')
- SVC() : SVM Classifier
- make_pipeline , StandardScaler() : Classifier에 사용할 학습 pipeline을 만들어주는 함수와 사용한 데이터의 정규 Scaling을 해주는 함수
- make_pipeline(StandardScaler(),BalancedRandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced', random_state=0, sampling_strategy = 'all'))
- make_pipeline(StandardScaler(), SVC(gamma='auto'))
