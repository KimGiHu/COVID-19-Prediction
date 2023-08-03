# import the necessary packages
import pandas as pd 
import numpy as np
import os
import spacy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='root dir for data')
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
# reference : https://www.programiz.com/python-programming/datetime/strftime
# strftime :  method returns a string representing date and time using date, time or datetime object.
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

print(f'dates[0] = {dates[0]}')

# set the train,validation,test datasets
train_dates = pd.date_range(start='2019-08-01', end='2022-08-31', freq='D').tolist()
val_dates = pd.date_range(start='2022-09-01', end='2022-11-30', freq='D').tolist() 
test_dates = pd.date_range(start='2022-12-01', end='2023-02-28', freq='D').tolist() 

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(val_dates)):
    val_dates[i] = val_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

print('dates', len(dates))
print('train_dates', len(train_dates))
print('val_dates', len(val_dates))
print('test_dates', len(test_dates))

# substract a list of 'USA' type dataset from the multi-language datasets
path = []
temp = os.listdir('./data_covid')
for item in temp :

    if args.lang in item :
        path.append(item)
path.sort()

# small model is not operated 
# so, i change the small model to the large model
nlps = {
    'korean': spacy.load('ko_core_news_lg'),
    'english': spacy.load('en_core_web_lg'),
    'japanese': spacy.load('ja_core_news_lg'),
    'french': spacy.load('fr_core_news_lg'),
    'italiano': spacy.load('it_core_news_lg'),
    'spanish': spacy.load('es_core_news_lg'),
    'russian': spacy.load('ru_core_news_lg'),
    'portuguese': spacy.load('pt_core_news_lg'),
}

# languages
languages = ['korean', 'english', 'japanese', 'french', 'italiano', 'spanish', 'russian', 'portuguese']

# language code
codes = {
    'korean': 'ko',
    'english': 'en',
    'japanese': 'ja',
    'french': 'fr',
    'italiano': 'it',
    'spanish': 'es',
    'russian': 'ru',
    'portuguese': 'pt',
}

# read stop_words
stop_words = {}
for language in languages:
    stop_words[language] = list(nlps[language].Defaults.stop_words)
    with open(f'../stopword/stopwords_{codes[language]}.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stop_words[language].extend(line.split())
    stop_words[language] = set(stop_words[language])

# tokenizer
def tokenizer(text, language):
    if language in languages:
        
        nlp = nlps[language]
        doc = nlp(text)
        words = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] if not token.is_stop and not token.is_punct and not token.is_oov and '+' not in token.lemma_]
        words = [word for word in words if word not in stop_words[language]]
        return words
    else:
        return text.split(" ")

#######################################
########### Tokenization ##############
#######################################

# to make language directory
createDirectory('./python_TF-IDF_%s'%args.lang)

for date in dates:
    print('processing %s' %date)
    temp = []
    for item in path : 
        if date in item :
            articles = pd.read_csv('./data_covid/%s'%item, lineterminator='\n')
            # pd.dropna() : dropna 메서드는 DataFramde내의 결측값이 포함된 레이블을 제거하는 메서드입니다.
            articles = articles['full_text'].dropna().tolist()
            temp.extend(articles)
    createDirectory('./python_TF-IDF_%s'%args.lang + '/data_kimgihu')
    pd.DataFrame({'full_text':temp}).to_csv('./python_TF-IDF_%s'%args.lang + '/data_kimgihu/%s.csv'%date, index=False)

select_lang = ''

# if the input language argument is Republic of Korea
if args.lang == 'Republic of Korea':
    select_lang = languages[0]
# if the input language argument is United States
if args.lang == 'United States':
    select_lang = languages[1]
# if the input language argument is Japan
if args.lang == 'Japan':
    select_lang = languages[2]
# if the input language argument is France
if args.lang == 'France':
    select_lang = languages[3]
# if the input language argument is Italy
if args.lang == 'Italy':
    select_lang = languages[4]
# if the input language argument is Russia
if args.lang == 'Russia':
    select_lang = languages[6]
# if the input language argument is Portugal
if args.lang == 'Portugal':
    select_lang = languages[7]

# to find the maximum index of len(file[i])
max = 10000
total_num = 0 
count = 0
print("find the max values")
for date in dates:
    file = pd.read_csv('./python_TF-IDF_%s'%args.lang +'/data_kimgihu/%s.csv'%date)['full_text'].tolist()
    for i in range(len(file)):
        total_num += 1
        if max < len(file[i]):
            # max = len(file[i]) 
            count += 1

print(f'max value of file[i] = {max}')
print(f'total_num = {total_num}')
print(f'count = {count}')

###################################################################################################
# normalizing and toknizing articles, removing stop word and saving the processed file using Spacy#
###################################################################################################

for date in dates:
    print('processing %s'%date)
    file = pd.read_csv('./python_TF-IDF_%s'%args.lang +'/data_kimgihu/%s.csv'%date)['full_text'].tolist()
    for i in range(len(file)):
        # only the legnth of file[i] is under 10000.
        if len(file[i]) <= 10000:
            file[i] = ' '.join(tokenizer(file[i], select_lang))
            
    pd.DataFrame({'full_text':file}).to_csv('./python_TF-IDF_%s'%args.lang +'/data_kimgihu/n_%s.csv'%date, index=False)

################################
######### set_words ############
################################

print("set_words")
set_words = []

for date in dates:
    print('processing %s'%date)
    file = pd.read_csv('./python_TF-IDF_%s'%args.lang +'/data_kimgihu/n_%s.csv'%date)['full_text'].tolist()
    for i in range(len(file)):
        newlist = file[i].split(' ')
        set_words.extend(newlist) 

set_words = set(set_words)

# print(f'set_words : {set_words}')
print(f'length of set_words : {len(set_words)}')

# creating an appendix of words and sorthing them by appearance frequency
total_text = []

print("total_text")
for date in dates:
    print(date)
    file = pd.read_csv('./python_TF-IDF_%s'%args.lang +'/data_kimgihu/n_%s.csv'%date)['full_text'].dropna().tolist()
    for article in file:
        total_text.extend(article.split(' '))

dict_word = {}
for index, item in enumerate(set_words):
    dict_word[item] = index
    
matrix_word = np.zeros(len(set_words))

for item in total_text:
    matrix_word[dict_word[item]] += 1
    
# sort by count in descending order
table = pd.DataFrame({'word':list(dict_word.keys()),'count':matrix_word}).sort_values(by='count', ascending=False)
# drop words that appear less than 100 times
# 100 -> 200 times [2023.07.24]
table = table[table['count'] > 200]
table = table.reset_index().drop(columns=['index'])
table.to_csv('./python_TF-IDF_%s'%args.lang +'/keywords_200.csv')

# key_date = pd.DataFrame(pd.read_csv('./python_TF-IDF/key_dates_US.csv'), columns=['date','label'])
# list_word = table['word'].values.tolist()


######################################################
##### vectorize the data of each date (ver.02)########
######################################################
keywords = pd.read_csv('./python_TF-IDF/keywords_200.csv')['word'].tolist()

# keywords: list
# searching: list.index(item)

for date in dates:
    print(date)
    file = pd.read_csv('./python_TF-IDF/data_kimgihu/n_%s.csv'%date)['full_text'].dropna().tolist()
    total_data = np.zeros(len(keywords), dtype=int)
    doc_vectors = []
    
    for i in range(len(file)):
        l = eval(file[i])
        doc_vector = np.zeros(len(keywords), dtype=int)
        
        for item in l:
            if item in keywords:
                doc_vector[keywords.index(item)] += 1
        total_data += doc_vector
        doc_vectors.append(doc_vector)
    
    doc_vectors = np.array(doc_vectors)
    np.savetxt('./python_TF-IDF/dvectors/%s.csv'%date, doc_vectors, fmt = '%d', delimiter=',')
    np.savetxt('./python_TF-IDF/vectors2/%s.csv'%date, total_data, fmt = '%d', delimiter=',')

doc_vectors = 0
   
for date in val_dates:
    doc_vector = np.loadtxt('./python_TF-IDF/dvectors/%s.csv'%date, delimiter=',')
    doc_vectors += len(doc_vector)

print(f'the number of validation vectors : {doc_vectors}')

doc_vectors = 0
   
for date in test_dates:
    doc_vector = np.loadtxt('./python_TF-IDF/dvectors/%s.csv'%date, delimiter=',')
    doc_vectors += len(doc_vector)

print(f'the number of test vectors : {doc_vectors}')

doc_vectors = 0
   
for date in dates:
    doc_vector = np.loadtxt('./python_TF-IDF/dvectors/%s.csv'%date, delimiter=',')
    doc_vectors += len(doc_vector)

print(f'the number of all vectors : {doc_vectors}')
