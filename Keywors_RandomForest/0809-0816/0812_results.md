# K-fold Corss valiation을 이용한 실험결과(K=5)

1. 영어 : 
  1) 키워드추출 결과
    2-1. Proposed Method
    rev_metric1 ['variant', 'coronavirus', 'covid', 'virus', 'pandemic', 'nose', 'mask', 'testing', 'contact', 'symptom', 'cough', 'throat', 'flu', 'individual', 'measure']
    rev_metric2 ['covid', 'coronavirus', 'pandemic', 'lockdown', 'variant', 'mask', 'vaccine', 'virus', 'datum', 'medication', 'therapy', 'fever', 'dose', 'time', 'people']
    rev_metric3 ['variant', 'coronavirus', 'covid', 'virus', 'pandemic', 'nose', 'mask', 'testing', 'contact', 'symptom', 'cough', 'throat', 'flu', 'individual', 'datum']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['people', 'time', 'patient', 'day', 'symptom', 'covid', 'health', 'disease', 'study', 'treatment', 'body', 'week', 'life', 'child', 'risk']
  2) 실험 결과
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.7017713147476469
    rev_metric1
    AUC for the rev_metric1 10 keywords  = 0.7541377946727813
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.7017713147476469
    rev_metric3
    AUC for the rev_metric3 10 keywords  = 0.7541377946727813
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.77 
    rev_metric1 0.82 
    rev_metric2 0.84 
    rev_metric3 0.82 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.77 
    rev_metric1 0.82 
    rev_metric2 0.84 
    rev_metric3 0.82

2. 한국어 :
  1) 키워드추출 결과
  2-1. Proposed Method
    rev_metric1 ['확진', '백신', '검사', '코로나', '증상', '치료', '발열', '접종', '수술', '발생', '신규', '임상', '환자', '해당', '대비']
    rev_metric2 ['코로나', '코로나바이러스', '감염증', '접종', '백신', '확진', '마스크', '비공감', '리얼타임', '공감', '방역', '증상', '감염', '환자', '치료']
    rev_metric3 ['확진', '백신', '검사', '코로나', '증상', '치료', '발열', '발생', '수술', '접종', '신규', '환자', '임상', '해당', '대비']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['코로나', '경우', '기자', '금지', '사진', '뉴스', '이후', '재배포', '최근', '지난해', '저작권자', '국내', '현재', '관련', '이날']
  2) 실험 결과
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.5407560621768991
    rev_metric1
    AUC for the rev_metric1 10 keywords  = 0.5632871760456077
    rev_metric1
    AUC for the rev_metric1 15 keywords  = 0.5753316984088621
    rev_metric1
    AUC for the rev_metric1 40 keywords  = 0.5774725476556767
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.5407560621768991
    rev_metric3
    AUC for the rev_metric3 10 keywords  = 0.5668439539930914
    rev_metric3
    AUC for the rev_metric3 45 keywords  = 0.56912047345057
    rev_metric3
    AUC for the rev_metric3 60 keywords  = 0.5697907872780948
    rev_metric3
    AUC for the rev_metric3 85 keywords  = 0.5811965633724465
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.63 
    rev_metric1 0.71 
    rev_metric2 0.63 
    rev_metric3 0.74 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.65 
    rev_metric1 0.62 
    rev_metric2 0.64 
    rev_metric3 0.64 
    
3. 일본어 : 
  1) 키워드추출 결과 
  2-1. Proposed Method
    rev_metric1 ['感染', '新型', '拡大', '接触', '発熱', '陽性', '検査', '確認', '株', '重症', '対策', '機関', '症状', '医療', '自宅']
    rev_metric2 ['禍', '陽性', '接種', '新型', '感染', '症', '薬', '人', '日', '年', '月', '者', '思', '中', '上']
    rev_metric3 ['感染', '新型', '拡大', '発熱', '接触', '陽性', '検査', '確認', '株', '重症', '対策', '症状', '機関', '医療', '自宅']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['日', '人', '年', '月', '者', '思', '感染', '中', '上', '性', '行', '大', '時', '感', '後']
  2) 실험 결과 
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.6181528772954572
    rev_metric1
    AUC for the rev_metric1 10 keywords  = 0.6308317031248315
    rev_metric1
    AUC for the rev_metric1 45 keywords  = 0.6319264875155179
    rev_metric1
    AUC for the rev_metric1 55 keywords  = 0.6373557710599924
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.6181528772954572
    rev_metric3
    AUC for the rev_metric3 10 keywords  = 0.6186155765078556
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.60 
    rev_metric1 0.71 
    rev_metric2 0.68 
    rev_metric3 0.65 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.64 
    rev_metric1 0.67 
    rev_metric2 0.65 
    rev_metric3 0.70
    
4. 프랑스어 :
  1) 키워드추출 결과 
    2-1. Proposed Method
    rev_metric1 ['mars', 'coronavirus', 'vaccin', 'symptôme', 'virus', 'masque', 'test', 'pandémie', 'huile', 'toux', 'janvier', 'infection', 'mesure', 'mardi', 'mercredi']
    rev_metric2 ['coronavirus', 'pandémie', 'confinement', 'vaccin', 'virus', 'symptôme', 'infection', 'an', 'jour', 'cas', 'heure', 'année', 'santé', 'enfant', 'vie']
    rev_metric3 ['mars', 'coronavirus', 'vaccin', 'symptôme', 'virus', 'masque', 'test', 'pandémie', 'janvier', 'infection', 'huile', 'toux', 'mesure', 'mardi', 'mercredi']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['an', 'jour', 'cas', 'heure', 'année', 'santé', 'enfant', 'vie', 'temps', 'maladie', 'femme', 'mois', 'patient', 'symptôme', 'monde']
  2) 실험 결과 
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.5732272927392998
    rev_metric1
    AUC for the rev_metric1 10 keywords  = 0.6762178558292289
    rev_metric1
    AUC for the rev_metric1 15 keywords  = 0.6835690228075777
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.5732272927392998
    rev_metric3
    AUC for the rev_metric3 10 keywords  = 0.6937385046742482
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.73 
    rev_metric1 0.63 
    rev_metric2 0.74 
    rev_metric3 0.65 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.72 
    rev_metric1 0.58 
    rev_metric2 0.71 
    rev_metric3 0.65 
    
5. 이탈리아어 :
  1) 키워드추출 결과
    2-1. Proposed Method
    rev_metric1 ['virus', 'coronavirus', 'sintomo', 'gravidanza', 'paziente', 'bambino', 'disturbo', 'tampone', 'sistema', 'dato', 'medico', 'marzo', 'situazione', 'salute', 'temperatura']
    rev_metric2 ['pandemia', 'coronavirus', 'virus', 'gatto', 'cane', 'sintomo', 'bambino', 'paziente', 'malattia', 'donna', 'causa', 'problema', 'punto', 'famiglia', 'medico']
    rev_metric3 ['virus', 'coronavirus', 'sintomo', 'gravidanza', 'paziente', 'bambino', 'disturbo', 'sistema', 'medico', 'dato', 'situazione', 'salute', 'marzo', 'temperatura', 'acqua']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['sintomo', 'bambino', 'paziente', 'malattia', 'donna', 'causa', 'problema', 'punto', 'medico', 'studio', 'famiglia', 'storia', 'figlio', 'rischio', 'dolore']
  2) 실험 결과
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.4904435720283942
    rev_metric1
    AUC for the rev_metric1 10 keywords  = 0.549819296640932
    rev_metric1
    AUC for the rev_metric1 50 keywords  = 0.5504435982851259
    rev_metric1
    AUC for the rev_metric1 75 keywords  = 0.5557789189833356
    rev_metric1
    AUC for the rev_metric1 80 keywords  = 0.5561696193223975
    rev_metric1
    AUC for the rev_metric1 85 keywords  = 0.5608427345086888
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.4904435720283942
    rev_metric3
    AUC for the rev_metric3 10 keywords  = 0.549819296640932
    rev_metric3
    AUC for the rev_metric3 75 keywords  = 0.5587765933056544
    rev_metric3
    AUC for the rev_metric3 90 keywords  = 0.5612031090723653
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.71 
    rev_metric1 0.70 
    rev_metric2 0.74 
    rev_metric3 0.67 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.73 
    rev_metric1 0.67 
    rev_metric2 0.70 
    rev_metric3 0.74
6. 러시아어 :
  1) 키워드추출 결과
    2-1. Proposed Method
    rev_metric1 ['пациент', 'симптом', 'заболевание', 'врач', 'больница', 'инфекция', 'боль', 'вирус', 'болезнь', 'температура', 'ребёнок', 'помощь', 'область', 'лечение', 'результат']
    rev_metric2 ['ребёнок', 'врач', 'заболевание', 'симптом', 'пациент', 'боль', 'помощь', 'болезнь', 'проблема', 'область', 'температура', 'состояние', 'организм', 'больница', 'инфекция']
    rev_metric3 ['пациент', 'симптом', 'врач', 'заболевание', 'боль', 'больница', 'инфекция', 'болезнь', 'вирус', 'температура', 'ребёнок', 'помощь', 'область', 'лечение', 'проблема']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['ребёнок', 'врач', 'заболевание', 'симптом', 'пациент', 'боль', 'болезнь', 'помощь', 'область', 'проблема', 'температура', 'больница', 'состояние', 'организм', 'лечение']
  2) 실험 결과
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.7362813301971577
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.7362813301971577
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.52 
    rev_metric1 0.60 
    rev_metric2 0.52 
    rev_metric3 0.60 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.52 
    rev_metric1 0.60 
    rev_metric2 0.52 
    rev_metric3 0.60 
    
7. 포르투갈어:
  1) 키워드추출 결과
    2-1. Proposed Method
    rev_metric1 ['coronavírus', 'vacina', 'pandemia', 'gripe', 'vírus', 'teste', 'variante', 'febre', 'atendimento', 'medida', 'dose', 'sintoma', 'tosse', 'máscara', 'infecção']
    rev_metric2 ['pandemia', 'coronavírus', 'vacina', 'vírus', 'pessoa', 'caso', 'doença', 'sintoma', 'dor', 'saúde', 'vida', 'paciente', 'feira', 'casa', 'corpo']
    rev_metric3 ['coronavírus', 'vacina', 'pandemia', 'vírus', 'gripe', 'teste', 'febre', 'atendimento', 'medida', 'sintoma', 'variante', 'dose', 'infecção', 'tosse', 'mão']
    2-2. Class-Based TF-IDF Baselines
    ctfidf ['pessoa', 'caso', 'doença', 'sintoma', 'dor', 'saúde', 'vida', 'paciente', 'feira', 'casa', 'corpo', 'tratamento', 'criança', 'morte', 'problema']
  2) 실험 결과
    rev_metric1
    AUC for the rev_metric1 5 keywords  = 0.5882479186240376
    rev_metric1
    AUC for the rev_metric1 10 keywords  = 0.6250813513713595
    rev_metric1
    AUC for the rev_metric1 15 keywords  = 0.6429018614035524
    rev_metric1
    AUC for the rev_metric1 60 keywords  = 0.642945119313741
    rev_metric1
    AUC for the rev_metric1 70 keywords  = 0.6497724149089955
    rev_metric1
    AUC for the rev_metric1 115 keywords  = 0.6524938704021945
    rev_metric1
    AUC for the rev_metric1 125 keywords  = 0.652729335590444
    rev_metric1
    AUC for the rev_metric1 135 keywords  = 0.6668008942568198
    rev_metric3
    AUC for the rev_metric3 5 keywords  = 0.5882479186240376
    rev_metric3
    AUC for the rev_metric3 10 keywords  = 0.6116420493809058
    rev_metric3
    AUC for the rev_metric3 15 keywords  = 0.634356787184913
    rev_metric3
    AUC for the rev_metric3 30 keywords  = 0.6455453765962242
    rev_metric3
    AUC for the rev_metric3 40 keywords  = 0.6478926694846247
    rev_metric3
    AUC for the rev_metric3 65 keywords  = 0.6504340697734365
    rev_metric3
    AUC for the rev_metric3 100 keywords  = 0.6557027301657898
    rev_metric3
    AUC for the rev_metric3 155 keywords  = 0.6557402674833241
    ######################## rev_metric1 AUC Results ########################
    Test AUC
    ctfidf 0.66 
    rev_metric1 0.60 
    rev_metric2 0.58 
    rev_metric3 0.61 
    ######################## rev_metric3 AUC Results ########################
    Test AUC
    ctfidf 0.58 
    rev_metric1 0.59 
    rev_metric2 0.64 
    rev_metric3 0.6

