# Code Explanation
## classifier
classifier_v4.py : Multi-Classifier (Standardized RF,SVC)
classifier_v5.py : Multi-Classifier (Standardized RF,SVC) + ReOverSampling
freq_multi_v3.py : Discirimitve Keyword Extraction Mehtod

- - - - - - 
# Results : country_verN.txt (Coutnry=Korea,UnitedStates,Japan / N=1,2,3,4)

나라별 재생산지수 RE Factor을 정답으로 이용하여 RF(RandomForest) / SVC(Support Vector machine Classifier) / Standardized RF / Standardized SVC 모델을 학습하는데 이용.
## Ver1

   make_pipeline(StandardScaler(),                     
                  BalancedRandomForestClassifier       
                  (n_estimators=100, max_depth=50,     
                  n_jobs=-1,                           
                  class_weight='balanced',             
                  random_state=0,                      
                  sampling_strategy = 'all')           
                  )                                    


## Ver2

 make_pipeline(StandardScaler(), SVC(gamma='auto')) 


## Ver3

 ros = RandomOverSampler(random_state=0)            
 make_pipeline(StandardScaler(), SVC(gamma='auto')) 


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
                  
