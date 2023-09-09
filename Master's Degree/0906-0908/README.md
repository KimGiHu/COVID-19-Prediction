# 09월 6일 ~ 09월 8일 실험결과 

0. 전처리한 결과 :

- **NOUN_VERB_ADJ**

total vocab size :  521296

- **NOUN_VERB**
  
total vocab size :  521267

### 1. 기존의 키워드 추출방법 : 

**NOUN rev_metric1 F1-score : 0.78  /   index : 25.00**

**VERB rev_metric1 F1-score : 0.78  /   index : 15.00**

**ADJ rev_metric1 F1-score : 0.78  /   index : 5.00**

### 2. 새로운 키워드 추출방법
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- **동사단위 키워드** 

new_metric1 ['lead', 'experience', 'feel', 'include', 'increase', 'affect', 'occur', 'lose', 'eat', 'leave', 'reduce', 'improve', 'add', 'choose', 'relate']

new_metric2 ['wash', 'stay', 'spread', 'cough', 'touch', 'close', 'kill', 'travel', 'avoid', 'isolate', 'confirm', 'sneeze', 'quarantine', 'cancel', 'contract']

new_metric3 ['experience', 'fall', 'occur', 'feel', 'boost', 'affect', 'breathe', 'score', 'prescribe', 'contact', 'opt', 'relieve', 'walk', 'sell', 'enrich']

new_metric4 ['include', 'receive', 'wear', 'develop', 'stay', 'write', 'provide', 'create', 'report', 'remain', 'start', 'protect', 'wash', 'associate', 'serve']

new_metric5 ['include', 'lead', 'increase', 'develop', 'feel', 'experience', 'reduce', 'receive', 'lose', 'base', 'report', 'affect', 'treat', 'follow', 'start']

new_metric6 ['spread', 'wash', 'sneeze', 'confirm', 'touch', 'cough', 'infect', 'cancel', 'travel', 'close', 'kill', 'stay', 'isolate', 'avoid', 'sound']

ctfidf1 ['include', 'feel', 'accord', 'add', 'follow', 'lead', 'start', 'report', 'increase', 'leave', 'eat', 'continue', 'provide', 'experience', 'play']

ctfidf2 ['include', 'feel', 'accord', 'follow', 'add', 'start', 'lead', 'report', 'increase', 'leave', 'continue', 'eat', 'provide', 'receive', 'develop']

ctfidf3 ['include', 'feel', 'accord', 'follow', 'add', 'start', 'lead', 'report', 'leave', 'continue', 'eat', 'increase', 'provide', 'play', 'die']

######################## ctfidf1 F1-score Results ########################

ctfidf1 F1-score : 0.78 index : 15.00 

######################## ctfidf2 F1-score Results ########################

ctfidf2 F1-score : 0.78 index : 25.00 

######################## ctfidf3 F1-score Results ########################

ctfidf3 F1-score : 0.78 index : 15.00 

######################## new_metric1 F1-score Results ########################

new_metric1 F1-score : 0.78     index : 25.00 

######################## new_metric2 F1-score Results ########################

new_metric2 F1-score : 0.78     index : 20.00 

######################## new_metric3 F1-score Results ########################

new_metric3 F1-score : 0.78     index : 30.00 

######################## new_metric4 F1-score Results ########################

new_metric4 F1-score : 0.78     index : 20.00 

######################## new_metric5 F1-score Results ########################

new_metric5 F1-score : 0.78     index : 30.00 

######################## new_metric6 F1-score Results ########################

new_metric6 F1-score : 0.78     index : 15.00 


**######################## new_metric & multi-classed preprocessed datasets F1-score Results ########################**
**new_metric6 F1-score : 0.78     index : 10.00**
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

- **명사단위**

new_metric1 ['pain', 'heart', 'symptom', 'child', 'doctor', 'body', 'infection', 'patient', 'blood', 'condition', 'cancer', 'medication', 'treatment', 'headache', 'time']

new_metric2 ['coronavirus', 'virus', 'hand', 'outbreak', 'spread', 'flu', 'covid', 'people', 'distancing', 'store', 'contact', 'lockdown', 'official', 'water', 'country']

new_metric3 ['heart', 'pain', 'doctor', 'headache', 'wind', 'attack', 'snow', 'infection', 'child', 'symptom', 'variant', 'chest', 'medication', 'fungus', 'depression']

new_metric4 ['covid', 'vaccine', 'pandemic', 'coronavirus', 'patient', 'study', 'people', 'water', 'lockdown', 'trial', 'week', 'school', 'cell', 'virus', 'death']

new_metric5 ['patient', 'covid', 'symptom', 'study', 'vaccine', 'time', 'cancer', 'blood', 'child', 'disease', 'treatment', 'body', 'pain', 'weight', 'day']

new_metric6 ['coronavirus', 'virus', 'outbreak', 'hand', 'spread', 'variant', 'flu', 'official', 'sound', 'breath', 'cough', 'store', 'epidemic', 'contact', 'travel']

ctfidf1 ['people', 'time', 'day', 'symptom', 'patient', 'health', 'covid', 'disease', 'body', 'child', 'study', 'life', 'treatment', 'pain', 'week']

ctfidf2 ['people', 'time', 'patient', 'day', 'covid', 'symptom', 'health', 'disease', 'study', 'treatment', 'body', 'life', 'week', 'child', 'death']

ctfidf3 ['people', 'time', 'day', 'patient', 'symptom', 'covid', 'health', 'disease', 'study', 'week', 'life', 'treatment', 'body', 'death', 'coronavirus']

######################## ctfidf1 F1-score Results ########################

ctfidf1 F1-score : 0.77 index : 15.00 

######################## ctfidf2 F1-score Results ########################

ctfidf2 F1-score : 0.77 index : 10.00 

######################## ctfidf3 F1-score Results ########################

ctfidf3 F1-score : 0.77 index : 20.00 

######################## new_metric1 F1-score Results ########################

new_metric1 F1-score : 0.77     index : 15.00 

######################## new_metric2 F1-score Results ########################

new_metric2 F1-score : 0.77     index : 25.00 

######################## new_metric3 F1-score Results ########################

new_metric3 F1-score : 0.77     index : 10.00 

######################## new_metric4 F1-score Results ########################

new_metric4 F1-score : 0.77     index : 10.00 

######################## new_metric5 F1-score Results ########################

new_metric5 F1-score : 0.77     index : 5.00 

######################## new_metric6 F1-score Results ########################

new_metric6 F1-score : 0.77     index : 30.00 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- **형용사단위**

new_metric1 ['medical', 'common', 'professional', 'cold', 'physical', 'bad', 'severe', 'russian', 'healthy', 'mental', 'black', 'inflammatory', 'abdominal', 'rare', 'clinical']

new_metric2 ['sick', 'social', 'public', 'essential', 'virtual', 'global', 'federal', 'protective', 'economic', 'national', 'festive', 'italian', 'runny', 'infected', 'safe']

new_metric3 ['cold', 'common', 'professional', 'medical', 'chinese', 'visceral', 'visual', 'mild', 'black', 'wrong', 'abdominal', 'physical', 'cardiac', 'bad', 'fungal']

new_metric4 ['social', 'public', 'safe', 'virtual', 'essential', 'immune', 'clinical', 'white', 'federal', 'economic', 'sick', 'national', 'global', 'effective', 'advanced']

new_metric5 ['medical', 'severe', 'clinical', 'common', 'professional', 'healthy', 'physical', 'immune', 'bad', 'inflammatory', 'russian', 'mental', 'chronic', 'personal', 'positive']

new_metric6 ['sick', 'chinese', 'public', 'runny', 'olympic', 'infected', 'festive', 'bronchial', 'italian', 'vesicular', 'unvaccinated', 'deadly', 'contagious', 'ulcerative', 'elderly']

ctfidf1 ['medical', 'common', 'bad', 'late', 'severe', 'positive', 'healthy', 'social', 'cold', 'clinical', 'public', 'physical', 'local', 'black', 'hard']

ctfidf2 ['medical', 'common', 'severe', 'late', 'bad', 'positive', 'social', 'healthy', 'public', 'clinical', 'local', 'immune', 'safe', 'hard', 'physical']

ctfidf3 ['medical', 'common', 'social', 'late', 'severe', 'positive', 'bad', 'public', 'healthy', 'local', 'safe', 'clinical', 'respiratory', 'sick', 'hard']

######################## ctfidf1 F1-score Results ########################

ctfidf1 F1-score : 0.78 index : 10.00 

######################## ctfidf2 F1-score Results ########################

ctfidf2 F1-score : 0.78 index : 10.00 

######################## ctfidf3 F1-score Results ########################

ctfidf3 F1-score : 0.78 index : 10.00 

######################## new_metric1 F1-score Results ########################

new_metric1 F1-score : 0.78     index : 30.00 

######################## new_metric2 F1-score Results ########################

new_metric2 F1-score : 0.78     index : 20.00 

######################## new_metric3 F1-score Results ########################

new_metric3 F1-score : 0.78     index : 10.00 

######################## new_metric4 F1-score Results ########################

new_metric4 F1-score : 0.78     index : 30.00 

######################## new_metric5 F1-score Results ########################

new_metric5 F1-score : 0.78     index : 15.00 

######################## new_metric6 F1-score Results ########################

new_metric6 F1-score : 0.78     index : 5.00

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- **명사/동사단위**
  
2-1. Keywords Extraction
  
new_metric1 ['pain', 'heart', 'symptom', 'child', 'experience', 'doctor', 'body', 'infection', 'patient', 'sign', 'blood', 'condition', 'lead', 'headache', 'cancer']

new_metric2 ['coronavirus', 'virus', 'hand', 'spread', 'outbreak', 'flu', 'covid', 'people', 'cough', 'wash', 'travel', 'quarantine', 'stay', 'distancing', 'store']

new_metric3 ['heart', 'pain', 'doctor', 'headache', 'wind', 'snow', 'infection', 'attack', 'child', 'symptom', 'variant', 'chest', 'medication', 'fungus', 'sign']

new_metric4 ['covid', 'vaccine', 'pandemic', 'coronavirus', 'study', 'patient', 'people', 'water', 'include', 'lockdown', 'receive', 'trial', 'school', 'week', 'cell']

new_metric5 ['patient', 'covid', 'study', 'symptom', 'include', 'vaccine', 'time', 'cancer', 'blood', 'child', 'disease', 'treatment', 'body', 'experience', 'pain']

new_metric6 ['coronavirus', 'virus', 'spread', 'outbreak', 'hand', 'variant', 'cough', 'sound', 'travel', 'flu', 'quarantine', 'sneeze', 'wash', 'official', 'confirm']

ctfidf1 ['people', 'time', 'day', 'patient', 'symptom', 'include', 'health', 'covid', 'disease', 'study', 'body', 'child', 'report', 'life', 'feel']

ctfidf2 ['people', 'time', 'patient', 'day', 'include', 'symptom', 'covid', 'study', 'health', 'disease', 'report', 'treatment', 'body', 'life', 'week']

ctfidf3 ['people', 'time', 'day', 'patient', 'include', 'symptom', 'covid', 'health', 'disease', 'study', 'report', 'week', 'life', 'feel', 'treatment']

######################## ctfidf1 F1-score Results ########################
ctfidf1 F1-score : 0.78 index : 10.00 
######################## ctfidf2 F1-score Results ########################
ctfidf2 F1-score : 0.78 index : 15.00 
######################## ctfidf3 F1-score Results ########################
ctfidf3 F1-score : 0.78 index : 10.00 
######################## new_metric1 F1-score Results ########################
new_metric1 F1-score : 0.78     index : 25.00 
######################## new_metric2 F1-score Results ########################
new_metric2 F1-score : 0.78     index : 10.00 
######################## new_metric3 F1-score Results ########################
new_metric3 F1-score : 0.78     index : 25.00 
######################## new_metric4 F1-score Results ########################
new_metric4 F1-score : 0.78     index : 5.00 
######################## new_metric5 F1-score Results ########################
new_metric5 F1-score : 0.78     index : 25.00 
######################## new_metric6 F1-score Results ########################
new_metric6 F1-score : 0.78     index : 30.00
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- **명사/동사/형용사단위**

2-1. Keywords Extraction

new_metric1 ['pain', 'heart', 'symptom', 'child', 'experience', 'doctor', 'body', 'infection', 'patient', 'sign', 'blood', 'medical', 'lead', 'condition', 'headache']

new_metric2 ['coronavirus', 'virus', 'hand', 'spread', 'outbreak', 'flu', 'people', 'public', 'covid', 'cough', 'wash', 'travel', 'quarantine', 'sick', 'stay']

new_metric3 ['heart', 'pain', 'cold', 'doctor', 'headache', 'wind', 'snow', 'infection', 'attack', 'child', 'symptom', 'variant', 'chest', 'medication', 'fungus']

new_metric4 ['covid', 'pandemic', 'vaccine', 'coronavirus', 'patient', 'study', 'people', 'water', 'include', 'lockdown', 'receive', 'trial', 'school', 'week', 'cell']

new_metric5 ['patient', 'covid', 'study', 'symptom', 'include', 'vaccine', 'time', 'cancer', 'blood', 'child', 'disease', 'treatment', 'body', 'experience', 'pain']

new_metric6 ['coronavirus', 'virus', 'spread', 'outbreak', 'hand', 'variant', 'cough', 'sound', 'travel', 'flu', 'sick', 'quarantine', 'sneeze', 'wash', 'official']

ctfidf1 ['people', 'time', 'patient', 'day', 'symptom', 'include', 'health', 'covid', 'disease', 'study', 'body', 'child', 'report', 'life', 'feel']

ctfidf2 ['people', 'time', 'patient', 'day', 'include', 'covid', 'symptom', 'study', 'health', 'disease', 'report', 'treatment', 'body', 'life', 'week']

ctfidf3 ['people', 'time', 'patient', 'day', 'include', 'symptom', 'covid', 'health', 'disease', 'study', 'report', 'life', 'week', 'feel', 'treatment']

######################## ctfidf1 F1-score Results ########################

ctfidf1 F1-score : 0.77 index : 10.00 

######################## ctfidf2 F1-score Results ########################

ctfidf2 F1-score : 0.77 index : 10.00 

######################## ctfidf3 F1-score Results ########################

ctfidf3 F1-score : 0.77 index : 10.00 

######################## new_metric1 F1-score Results ########################

new_metric1 F1-score : 0.77     index : 30.00 

######################## new_metric2 F1-score Results ########################

new_metric2 F1-score : 0.77     index : 15.00 

######################## new_metric3 F1-score Results ########################

new_metric3 F1-score : 0.77     index : 20.00 

######################## new_metric4 F1-score Results ########################

new_metric4 F1-score : 0.77     index : 30.00 

######################## new_metric5 F1-score Results ########################

new_metric5 F1-score : 0.77     index : 5.00 

######################## new_metric6 F1-score Results ########################

new_metric6 F1-score : 0.77     index : 30.00 


**######################## new_metric & multi-classed preprocessed datasets F1-score Results ########################**
**new_metric6 F1-score : 0.78     index : 5.00**
