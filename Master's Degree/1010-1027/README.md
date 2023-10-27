# 10/27 Self Feedback Report
현재 연구방향의 문제점 정리

1. 재생산지수 Re 기반의 분류 모델을 학습함에 있어서, 정답 레이블을 결정하는 기준이 모호하여 각 나라마다 유효한 레이블링이 이루어지지 않음 
- Level 1/2/3을 나누는 기준이 나라마다 다른 것으로 예상됨.
- 모델을 학습하기 이전에, 핵심단어를 추출시에 정답 레이블링을 나라마다 다르게 해줄 필요가 있어보임
- 레이블의 기준을 수학적 모델로 새롭게 제시한다면 유의미한 논문의 기여(Contribution)를 만들 수 있을거라 판단됨.

2. Level 1/ Level 2를 분류하는 모델 + Level 2/ Level 3을 분류하는 모델 
- if문을 이용하여, 앙상블 기법을 구현.
- 각각의 모델을 합쳐서 다중-클래스 분류모델의 성능을 향상.

3. 특이점 탐지 모델 by Seungtae Kang
- 아웃라이어들을 감지(?) 탐지(?) 해내는 분류 모데을 학습하도록, 전처리부터 워드 추출방법까지 재설계하여 새로운 분류 시스템을 제안함.
- 예상되는 문제점 : 특이점 탐지가 잘 가능한지 알기 위해서 한번 이상의 학습이 이루어지기까지의 시간이 크게 소요됨 + 지도학습(Supervised Learning)에서 비지도학습(Unsupervised Learning)으로 학습 방향을 바꾼 것이기에 성능을 보장 가능한가에 대한 의문이 생김.

# 석사학위논문에서 사용한 기계학습 분류모델들 및 분류성능평가 기준
### 1. Machine Learning 사용한 기법들
- StandardScaler 함수 + RandomOverSampler 패키지함수를 활용하여 데이터 불균형(Data Imbalance)을 잡아줌
1) Support Vector Machine
  - decision function shape1 : one vs. one
  - decision function shape2 : one vs. the rest
2) BalancedRandomForest
3) GaussianProcessClassifier
  - one vs. one
  - one vs. the rest
4) GradientBoostingClassifier / AdaBoostClassifier : 하이퍼파라미터(Hyperparameter)는 기본값을 사용함. 학습한 결과가 앞서 언급한 분류모델들보다 좋지 못함.
### 2. 분류성능평가 기준
- 이진분류(Binary-Classification) 모델 --> 다중분류(Multi-Classification) 모델
- 다중분류모델에서는 micro-Average F1 점수를 분류성능 기준으로 정함.
- micro-Avg를 계산하기 위해서는 Recall값과 Precision값 Accuracy값이 모두 동일한 값으로 잡히기에, **micro-Avg F1 score = Accruacy** 가 성립한다.
