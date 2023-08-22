# 성능평가를 위한 코드들
1. 7개의 국가 뉴스들에 대해 각각 전처리함
2. 전처리한 뉴스 DB들과 keydate가 1인 날짜들로부터 relative-frequency matrix를 생성함. (확진자수 증가하였다고 판단하고 이를 주요 keywords라고 생각.)
3. 생성한 relative freq. matrix와 keydate를 이용하여 확진자 증가일을 분류해내는 모델을 학습함.
4. 앞서 학습한 모델과 relative freq. matrix를 이용하여, 학습이 잘 되었는지를 평가함.(inference)
