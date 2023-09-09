09월 6일 ~ 09월 8일 실험결과 :

1. 기존의 키워드 추출방법
<United States>
rev_metric1 F1-score : 0.78     index : 15.00

2. 새로운 키워드 추출방법
1) 동사단위

 2-1. Keywords Extraction
 
new_metric1 ['lead', 'experience', 'feel', 'include', 'increase', 'affect', 'occur', 'lose', 'eat', 'leave', 'reduce', 'improve', 'add', 'choose', 'relate']

new_metric2 ['wash', 'stay', 'spread', 'cough', 'touch', 'close', 'kill', 'travel', 'avoid', 'isolate', 'confirm', 'sneeze', 'quarantine', 'cancel', 'contract']

new_metric3 ['experience', 'fall', 'occur', 'feel', 'boost', 'affect', 'breathe', 'score', 'prescribe', 'contact', 'opt', 'relieve', 'walk', 'sell', 'enrich']

new_metric4 ['include', 'receive', 'wear', 'develop', 'stay', 'write', 'provide', 'create', 'report', 'remain', 'start', 'protect', 'wash', 'associate', 'serve']

new_metric5 ['include', 'lead', 'increase', 'develop', 'feel', 'experience', 'reduce', 'receive', 'lose', 'base', 'report', 'affect', 'treat', 'follow', 'start']

new_metric6 ['spread', 'wash', 'sneeze', 'confirm', 'touch', 'cough', 'infect', 'cancel', 'travel', 'close', 'kill', 'stay', 'isolate', 'avoid', 'sound']

ctfidf1 ['include', 'feel', 'accord', 'add', 'follow', 'lead', 'start', 'report', 'increase', 'leave', 'eat', 'continue', 'provide', 'experience', 'play']

ctfidf2 ['include', 'feel', 'accord', 'follow', 'add', 'start', 'lead', 'report', 'increase', 'leave', 'continue', 'eat', 'provide', 'receive', 'develop']

ctfidf3 ['include', 'feel', 'accord', 'follow', 'add', 'start', 'lead', 'report', 'leave', 'continue', 'eat', 'increase', 'provide', 'play', 'die']

2) 명사단위

2-1. Keywords Extraction

new_metric1 ['pain', 'heart', 'symptom', 'child', 'doctor', 'body', 'infection', 'patient', 'blood', 'condition', 'cancer', 'medication', 'treatment', 'headache', 'time']

new_metric2 ['coronavirus', 'virus', 'hand', 'outbreak', 'spread', 'flu', 'covid', 'people', 'distancing', 'store', 'contact', 'lockdown', 'official', 'water', 'country']

new_metric3 ['heart', 'pain', 'doctor', 'headache', 'wind', 'attack', 'snow', 'infection', 'child', 'symptom', 'variant', 'chest', 'medication', 'fungus', 'depression']

new_metric4 ['covid', 'vaccine', 'pandemic', 'coronavirus', 'patient', 'study', 'people', 'water', 'lockdown', 'trial', 'week', 'school', 'cell', 'virus', 'death']

new_metric5 ['patient', 'covid', 'symptom', 'study', 'vaccine', 'time', 'cancer', 'blood', 'child', 'disease', 'treatment', 'body', 'pain', 'weight', 'day']

new_metric6 ['coronavirus', 'virus', 'outbreak', 'hand', 'spread', 'variant', 'flu', 'official', 'sound', 'breath', 'cough', 'store', 'epidemic', 'contact', 'travel']

ctfidf1 ['people', 'time', 'day', 'symptom', 'patient', 'health', 'covid', 'disease', 'body', 'child', 'study', 'life', 'treatment', 'pain', 'week']

ctfidf2 ['people', 'time', 'patient', 'day', 'covid', 'symptom', 'health', 'disease', 'study', 'treatment', 'body', 'life', 'week', 'child', 'death']

ctfidf3 ['people', 'time', 'day', 'patient', 'symptom', 'covid', 'health', 'disease', 'study', 'week', 'life', 'treatment', 'body', 'death', 'coronavirus']


3) 형용사단위

2-1. Keywords Extraction

new_metric1 ['medical', 'common', 'professional', 'cold', 'physical', 'bad', 'severe', 'russian', 'healthy', 'mental', 'black', 'inflammatory', 'abdominal', 'rare', 'clinical']

new_metric2 ['sick', 'social', 'public', 'essential', 'virtual', 'global', 'federal', 'protective', 'economic', 'national', 'festive', 'italian', 'runny', 'infected', 'safe']

new_metric3 ['cold', 'common', 'professional', 'medical', 'chinese', 'visceral', 'visual', 'mild', 'black', 'wrong', 'abdominal', 'physical', 'cardiac', 'bad', 'fungal']

new_metric4 ['social', 'public', 'safe', 'virtual', 'essential', 'immune', 'clinical', 'white', 'federal', 'economic', 'sick', 'national', 'global', 'effective', 'advanced']

new_metric5 ['medical', 'severe', 'clinical', 'common', 'professional', 'healthy', 'physical', 'immune', 'bad', 'inflammatory', 'russian', 'mental', 'chronic', 'personal', 'positive']

new_metric6 ['sick', 'chinese', 'public', 'runny', 'olympic', 'infected', 'festive', 'bronchial', 'italian', 'vesicular', 'unvaccinated', 'deadly', 'contagious', 'ulcerative', 'elderly']

ctfidf1 ['medical', 'common', 'bad', 'late', 'severe', 'positive', 'healthy', 'social', 'cold', 'clinical', 'public', 'physical', 'local', 'black', 'hard']

ctfidf2 ['medical', 'common', 'severe', 'late', 'bad', 'positive', 'social', 'healthy', 'public', 'clinical', 'local', 'immune', 'safe', 'hard', 'physical']

ctfidf3 ['medical', 'common', 'social', 'late', 'severe', 'positive', 'bad', 'public', 'healthy', 'local', 'safe', 'clinical', 'respiratory', 'sick', 'hard']

