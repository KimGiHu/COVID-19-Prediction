지금까지의 실험 결과 및 성과들을 저장해 놓기 위한 디렉토리입니다.

모든 실험 및 결과들은 다음과 같은 기간동안 진행되었습니다.

# ICCE-Asia 2023 Conference Paper & Simulation Results
- 주제 : Prediction of Increase in COVID-19 confirmed cases through multilingual keyword extraction
- 저자 : Gi-Hu Kim, Gil-Jin Jang*
- Abstract :

This paper proposes new method for COVID-19 spread prediction using the multi-lingual news. 
A novel keyword extraction method is proposed and a random forest classifier that takes the relative frequency of keyword appearances as input. 
According to the experimental results, the average test AUC(area under the ROC curve) scores of United States, Republic of Korea, and Japan showed 7.67% improved performance on the average compared to the existing system.

- proposed method :
 
  $RTCKD_i$= $\frac{TC_{i}^{kd}}{∑_{j}^{|D|}∑_{k}^{|T_{j} |}n_{k, j}}≈P(t_{i} |D^{(kd)})$
  
  
- 영어, 한국어, 일본어에 대해서 실험한 결과

|  Test AUC |	United States	| Japan	| Korea |
|-----------|---------------|-------|-------|
| NTFKD	| 0.79(C=20)	| 0.73(C=25)	| 0.54(C=10)|
| RTCKD	| 0.81(C=25)	| 0.79(C=25)	| 0.69(C=5)|

