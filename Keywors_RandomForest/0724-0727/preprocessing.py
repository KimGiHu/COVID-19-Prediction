# import the necessary packages
import pandas as pd 
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords

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
# print(os.getcwd())
print(os.listdir('./data_covid'))

temp = os.listdir('./data_covid')
for item in temp :
    if 'United States' in item :
        path.append(item)
path.sort()
print(path)


# for date in dates:
#     print('processing %s' %date)
#     temp = []
#     for item in path : 
#         if date in item :
#             articles = pd.read_csv('./data_covid/%s'%item, lineterminator='\n')
#             # pd.dropna() : dropna 메서드는 DataFramde내의 결측값이 포함된 레이블을 제거하는 메서드입니다.
#             articles = articles['full_text'].dropna().tolist()
#             temp.extend(articles)
#     pd.DataFrame({'full_text':temp}).to_csv('./python_TF-IDF/data_kimgihu/%s.csv'%date, index=False)

# # normalizing and toknizing articles, removing stop word and saving the processed file
# # re library : 수식하나로 특수문자를 제거하는 규칙 표현이 가능하게 해주는 라이브러리
# for date in dates:
#     print('processing %s'%date)
#     file = pd.read_csv('./python_TF-IDF/data_kimgihu/%s.csv'%date)['full_text'].tolist()
#     for i in range(len(file)):
#         newlist = []
#         text = re.sub(r'[^\w\s]',' ',file[i]) # 알파벳,숫자,_ 그리고 space를 sub함.
#         text = re.sub(r'[0-9]','',text) # 숫자 0-9를 sub함.
#         text = nltk.word_tokenize(text)
#         for item in text:
#             if item.lower() not in stopwords.words('english'):
#                 newlist.append(item.lower())
#         file[i] = newlist
#     pd.DataFrame({'full_text':file}).to_csv('./python_TF-IDF/data_kimgihu/n_%s.csv'%date, index=False)

# print("set_words")
# set_words = []
# for date in dates:
    
#     print('processing %s'%date)
#     file = pd.read_csv('./python_TF-IDF/data_kimgihu/n_%s.csv'%date)['full_text'].tolist()
#     for i in range(len(file)):
#         # file에서 읽어오는 데이터들은 expression형태로 저장이 되어 있기 때문에 다시 string 형태로 바꾸어줘야한다.
#         newlist = eval(file[i]) # eval : expression(=식)을 문자열로 받아와서, 실행하는 함수
#         set_words.extend(newlist) # extend : list 안의 있는 원소들 각각을 append 해주는 함수

# set_words = set(set_words)

# print(f'set_words : {set_words}')
# print(f'length of set_words : {len(set_words)}')

# # creating an appendix of words and sorthing them by appearance frequency
# total_text = []

# print("total_text")
# for date in dates:
#     print(date)
#     file = pd.read_csv('./python_TF-IDF/data_kimgihu/n_%s.csv'%date)['full_text'].dropna().tolist()
#     for article in file:
#         total_text.extend(eval(article))

# dict_word = {}
# for index, item in enumerate(set_words):
#     dict_word[item] = index
    
# matrix_word = np.zeros(len(set_words))

# for item in total_text:
#     matrix_word[dict_word[item]] += 1
    
# # sort by count in descending order
# table = pd.DataFrame({'word':list(dict_word.keys()),'count':matrix_word}).sort_values(by='count', ascending=False)
# # drop words that appear less than 100 times
# # 100 -> 200 times [2023.07.24]
# table = table[table['count'] > 200]
# table = table.reset_index().drop(columns=['index'])
# table.to_csv('./python_TF-IDF/keywords_200.csv')

# key_date = pd.DataFrame(pd.read_csv('./python_TF-IDF/key_dates_US.csv'), columns=['date','label'])
# list_word = table['word'].values.tolist()


# # keywords: list
# # searching: dict[item] = index, list[index]++


# dict_word = {}
# for index, item in enumerate(list_word):
#     dict_word[item] = index
# print("keyword")
# # 날짜에 나온 뉴스들만 계산함.
# for date in dates:
#     total_vector = np.zeros(len(list_word))
#     print('processing %s'%date)
#     file = pd.read_csv('./python_TF-IDF/data_kimgihu/n_%s.csv'%date)['full_text'].dropna().tolist()
#     for i in range(len(file)):
#         l = eval(file[i])
#         doc_vector = np.zeros(len(list_word))
#         for item in l:
#             if item in list_word:
#                 doc_vector[dict_word[item]] += 1
#         doc_vector = np.array(doc_vector, dtype=int)
#         total_vector += doc_vector
#     np.savetxt('./python_TF-IDF/vectors/%s.csv'%date, total_vector, fmt = '%d', delimiter=',')

    # vectorize the data of each date (ver.02)
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