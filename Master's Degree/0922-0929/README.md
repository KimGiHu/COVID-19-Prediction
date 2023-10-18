# 나라별 재생산지수 RE Factor을 정답으로 이용하여, RF(RandomForest) / SVC(Support Vector machine Classifier) / Standardized RF / Standardized SVC 모델을 학습하는데 이용.
## Ver1

   make_pipeline(StandardScaler(),                     
                  BalancedRandomForestClassifier       
                  (n_estimators=100, max_depth=50,     
                  n_jobs=-1,                           
                  class_weight='balanced',             
                  random_state=0,                      
                  sampling_strategy = 'all')           
                  )                                    

- - - - - - 
## Ver2

 make_pipeline(StandardScaler(), SVC(gamma='auto')) 

- - - - - - 
## Ver3

 ros = RandomOverSampler(random_state=0)            
 make_pipeline(StandardScaler(), SVC(gamma='auto')) 

- - - - - - 
## Ver4

   ros = RandomOverSampler(random_state=0)            
   make_pipeline(StandardScaler(),                     
                  BalancedRandomForestClassifier       
                  (n_estimators=100, max_depth=50,     
                  n_jobs=-1,                           
                  class_weight='balanced',             
                  random_state=0,                      
                  sampling_strategy = 'all')           
                  )                                  
                  
