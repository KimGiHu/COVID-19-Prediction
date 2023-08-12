# 진행상황
1. 다국어 지원 자연어처리 라이브러리 spaCy에서 제공하는 뉴스 대형 언어 모델(news large language model)의 기능과 정규식 표현(Regular Expression)을 이용한 필터링을 통해 각 나라별 뉴스기사로부터 해당 나라의 언어만을 추출하였다.

2. 추후에 실험을 진행한 결과를 비교하고, 이를 논문의 주요 기여도로 활용하였다.

3. K-fold Cross Validation(K=5)을 이용하여 hyper parameter을 찾고, vocab vector의 크기를 해당날짜의 뉴스길이만큼 나누어서 뉴스갯수에 의존적이지 않게끔, 뉴스 한개당 vocab vector을 따로 만들었다.


