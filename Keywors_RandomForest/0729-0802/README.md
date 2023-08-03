# 2023/08/02 수정사항
### 1. Spacy tokenizer 수정 
- regular Experession : korean [^ㄱ-ㅎㅏ-ㅣ가-힣] english [^a-zA-Z] Japanese [^ぁ-んァ-ン一-龯] 추가.

### 2. vocab size를 입력으로 받음
- korean / Japanese count : 100
- English count : 100

### 3. frequency matrix : 상위 300개까지 보던 것을 2000개로 수정함.
