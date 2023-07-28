# import the necessary packages
import pandas as pd 
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords

""" collect a list of dates """
# to return a fixed frequency DatetimeIndex. 'D' meaning : calendar day frequency
# dates = pd.date_range(start='2019-08-01', end='2020-07-31', freq='D')
# print(f'the data type of variable dates = {type(dates)}')
dates = pd.date_range(start='2019-08-01', end='2022-02-28', freq='D').tolist()
print(f'the data type of variable dates = {type(dates)}')
print(f'dates[0] = {dates[0]}')

# reference : https://www.programiz.com/python-programming/datetime/strftime
# strftime :  method returns a string representing date and time using date, time or datetime object.
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

print(f'dates[0] = {dates[0]}')


train_dates = pd.date_range(start='2019-08-01', end='2022-08-31', freq='D').tolist()
val_dates = pd.date_range(start='2022-09-01', end='2022-11-30', freq='D').tolist() 
test_dates = pd.date_range(start='2022-12-01', end='2023-02-28', freq='D').tolist() 

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(val_dates)):
    val_dates[i] = val_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

# substract a list of 'USA' type dataset from the multi-language datasets
path = []
# print(os.getcwd())
print(os.listdir('./data_covid'))

temp = os.listdir('./data_covid')
for item in temp :
    if 'United States' in item :
        path.append(item)
path.sort()


# to redefine the date times for stretch the training period
dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

keywords = pd.read_csv('./TF-IDF_Kimgihu/keywords.csv')['word'].tolist()

""" 1. seperate into key and non-key groups """
""" 2. calculate the word importance metric from the vectors """
key_vector = []
non_key_vector = []
   
key_date = pd.DataFrame(pd.read_csv('./TF-IDF_Kimgihu/key_dates_US.csv'), columns=['date','label'])

for date in train_dates:
    vector = np.loadtxt('./TF-IDF_Kimgihu/vectors2/%s.csv'%date, delimiter=',')
    if date in key_date[key_date['label']==1]['date'].tolist():
        key_vector.append(vector)
    else:
        non_key_vector.append(vector)

average_key = np.average(key_vector, axis=0)
average_non_key = np.average(non_key_vector, axis=0)

appearance_key = np.sum(np.array(key_vector) > 0, axis=0)
appearance_non_key = np.sum(np.array(non_key_vector) > 0, axis=0)

proportion_key = np.sum(key_vector, axis=0) / np.sum(np.array(key_vector))
proportion_non_key = np.sum(non_key_vector, axis=0) / np.sum(np.array(non_key_vector))

### soykeyword variation description
# metric1: proportion-based, divided the number of times a keyword appears in a document set by the total number of words in the document set
# metric2: appearance-based, counted once per document, no additional calculation
# metric3: average-based, divided the number of times a keyword appears in a document set by the total number of documents in the document set
soykeyword_metric1 = proportion_key/(proportion_key+proportion_non_key)
soykeyword_metric2 = appearance_key/(appearance_key+appearance_non_key)
soykeyword_metric3 = average_key / (average_key + average_non_key)

key_freq = np.sum(key_vector, axis=0)
non_key_freq = np.sum(non_key_vector, axis=0)
total_freq = key_freq + non_key_freq
freq_metric = np.tanh(30000 * key_freq / np.sum(total_freq))
rev_metric1 = soykeyword_metric1 * freq_metric
rev_metric2 = soykeyword_metric2 * freq_metric
rev_metric3 = soykeyword_metric3 * freq_metric

words = pd.read_csv('./TF-IDF_Kimgihu/keywords.csv')['word'].tolist()

data_metric = {'keywords':words,
               'average_key':average_key, 'average_non_key':average_non_key,
               'freq_key':np.sum(key_vector, axis=0), 'freq_non_key':np.sum(non_key_vector, axis=0),
               'total_freq': total_freq, 'freq_metric':freq_metric, 'rev_metric1':rev_metric1, 'rev_metric2':rev_metric2, 'rev_metric3':rev_metric3,
               'soykeyword_metric1':soykeyword_metric1, 'soykeyword_metric2':soykeyword_metric2, 'soykeyword_metric3':soykeyword_metric3}

pd.DataFrame(data_metric).to_csv('./TF-IDF_Kimgihu/metric_for_paper.csv', index=False)

# to form the total matrix of dates : keyword
complete_matrix = {}

for date in dates:
    vector = pd.read_csv('./TF-IDF_Kimgihu/vectors2/%s.csv'%date, header=None).values.flatten()
    complete_matrix[date] = vector

complete_table = pd.DataFrame(complete_matrix, index=keywords)
complete_table.to_csv('./TF-IDF_Kimgihu/complete_table.csv')

# to pick out the top keywords from the complete matrix
# to store the highest value of the 'named' metrics in dates
names = ['soykeyword_metric1', 'soykeyword_metric2', 'soykeyword_metric3', 'average_key', 'average_non_key', 'freq_key', 'freq_non_key', 'total_freq', 'freq_metric', 'rev_metric1', 'rev_metric2', 'rev_metric3']
for name in names:
    data_metric = pd.DataFrame(pd.read_csv('./TF-IDF_Kimgihu/metric_for_paper.csv'))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = data_metric['keywords'].tolist()
    pd.DataFrame(keywords).to_csv('./TF-IDF_Kimgihu/keywords/%s.csv'%name, index=False, header=False)
    total_data = pd.read_csv('./TF-IDF_Kimgihu/complete_table.csv', index_col='Unnamed: 0').T[keywords[:300]]
    total_data.to_csv('./TF-IDF_Kimgihu/frequency_matrix_%s.csv'%name)

## 2. TF-IDF based Baselines
key_vector = []
non_key_vector = []
key_doc_vectors = []
non_key_doc_vectors = []
   
key_date = pd.DataFrame(pd.read_csv('./TF-IDF_Kimgihu/key_dates_US.csv'), columns=['date','label'])

for date in train_dates:
    vector = np.loadtxt('./TF-IDF_Kimgihu/vectors2/%s.csv'%date, delimiter=',')
    doc_vectors = np.loadtxt('./TF-IDF_Kimgihu/dvectors/%s.csv'%date, delimiter=',')
    if date in key_date[key_date['label']==1]['date'].tolist():
        key_vector.append(vector)
        key_doc_vectors.append(doc_vectors)
    else:
        non_key_vector.append(vector)
        non_key_doc_vectors.append(doc_vectors)

key_vector = np.array(key_vector)
key_doc_vectors = np.concatenate(key_doc_vectors, axis=0)

non_key_vector = np.array(non_key_vector)
non_key_doc_vectors = np.concatenate(non_key_doc_vectors, axis=0)

print(key_vector.shape, key_doc_vectors.shape, non_key_vector.shape, non_key_doc_vectors.shape)


BTF = np.sum(key_vector, axis=0)
NTF1 = BTF/(np.max(BTF) + 1e-8)
NTF2 = np.sum(key_doc_vectors/(np.sum(key_doc_vectors, axis=1, keepdims=True) + 1e-8), axis=0)

print(BTF.shape, NTF1.shape, NTF2.shape)
print(np.sum(key_doc_vectors, axis=1, keepdims=True).shape)

IDF = np.log((len(key_doc_vectors))/(np.array([np.sum(key_doc_vectors[:, i] != 0) for i in range(len(key_vector[0]))])+1e-8) + 1e-8)
print(IDF.shape)
print(np.sum(key_doc_vectors[:, 0] != 0))

baseline1 = BTF*IDF
baseline2 = NTF1*IDF
baseline3 = NTF2*IDF
baseline4 = np.log(1 + BTF)*IDF
baseline5 = np.log(1 + NTF1)*IDF
baseline6 = np.log(1 + NTF2)*IDF

words = pd.read_csv('./TF-IDF_Kimgihu/keywords.csv')['word'].tolist()

data_metric = {'keywords':words,
               'BTF':BTF,
               'NTF1':NTF1,
               'NTF2':NTF2,
               'IDF':IDF,
               'baseline1':baseline1,
               'baseline2':baseline2,
               'baseline3':baseline3,
               'baseline4':baseline4,
               'baseline5':baseline5,
               'baseline6':baseline6}

pd.DataFrame(data_metric).to_csv('./TF-IDF_Kimgihu/metric_for_paper_tfidf.csv', index=False)

for name in ['baseline1', 'baseline2', 'baseline3', 'baseline4', 'baseline5', 'baseline6', 'BTF', 'NTF1', 'NTF2', 'IDF']:
    data_metric = pd.DataFrame(pd.read_csv('./TF-IDF_Kimgihu/metric_for_paper_tfidf.csv'))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = pd.read_csv('./TF-IDF_Kimgihu/keywords.csv')['word'].tolist()
    keywords = [keywords[i] for i in data_metric.index.tolist()]
    pd.DataFrame(keywords).to_csv('./TF-IDF_Kimgihu/keywords/%s.csv'%name, index=False, header=False)

    total_data = pd.read_csv('./TF-IDF_Kimgihu/complete_table.csv', index_col='Unnamed: 0').T[keywords[:300]]
    total_data.to_csv('./TF-IDF_Kimgihu/frequency_matrix_%s.csv' % name)

    print(name, total_data.columns.tolist()[:15])

## 2-2. Class-Based TF-IDF Based Baselines
tf_tc = np.sum(key_vector, axis=0)
tf_t = np.sum(key_vector, axis=0) + np.sum(non_key_vector, axis=0)

A = ( np.sum(key_vector) + np.sum(non_key_vector) ) / 2

ctfidf = tf_tc * np.log( 1 + (A / (tf_t + 1e-8)) )

tf_tc = np.sum(key_vector, axis=0)
tf_t = np.sum(key_vector, axis=0) + np.sum(non_key_vector, axis=0)

A = ( np.sum(key_vector) + np.sum(non_key_vector) ) / 2

ctfidf = tf_tc * np.log( 1 + (A / (tf_t + 1e-8)) )

words = pd.read_csv('./TF-IDF_Kimgihu/keywords.csv')['word'].tolist()

data_metric = {'keywords':words,
               'tf_tc':tf_tc,
               'tf_t':tf_t,
               'ctfidf':ctfidf
               }

pd.DataFrame(data_metric).to_csv('./TF-IDF_Kimgihu/metric_for_paper_ctfidf.csv', index=False)

for name in ['ctfidf']:
    data_metric = pd.DataFrame(pd.read_csv('./TF-IDF_Kimgihu/metric_for_paper_ctfidf.csv'))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = pd.read_csv('./TF-IDF_Kimgihu/keywords.csv')['word'].tolist()
    keywords = [keywords[i] for i in data_metric.index.tolist()]
    pd.DataFrame(keywords).to_csv('./TF-IDF_Kimgihu/keywords/%s.csv'%name, index=False, header=False)

    total_data = pd.read_csv('./TF-IDF_Kimgihu/complete_table.csv', index_col='Unnamed: 0').T[keywords[:300]]
    total_data.to_csv('./TF-IDF_Kimgihu/frequency_matrix_%s.csv' % name)

    print(name, total_data.columns.tolist()[:15])